using AutoBZ

# h, Σ, and β are fixed parameters
# μ is an interpolation parameter
function density_solver(; kws...)
    (; model, selfenergy, choose_kω_order, atol_n, rtol_n, quad_n_ω, quad_n_k, nworkers) = merge(default, NamedTuple(kws))

    h, bz, info_model = model(; kws...)
    Σ, info_selfenergy = selfenergy(; kws...)
    β = invtemp(; kws...)
    is_order_kω = choose_kω_order(quad_n_k, quad_n_ω)
    info = (; model=info_model, selfenergy=info_selfenergy, β, quad_n_ω, quad_n_k, is_order_kω, atol_n, rtol_n)

    w = AutoBZCore.workspace_allocate_vec(h, AutoBZCore.period(h), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
    ρ = if !is_order_kω
        # inner BZ integral
        f = AutoBZ.parentseries(h)
        bandwidth_bound = sqrt(sum(norm(c)^2 for c in f.c) - norm(f.c[-CartesianIndex(f.o)])^2)
        integrand = ElectronDensityIntegrand(bz, quad_n_k, w; Σ, β, abstol=atol_n*det(bz.B)/bandwidth_bound, reltol=rtol_n)
        IntegralSolver(integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), quad_n_ω; abstol=atol_n*det(bz.B), reltol=rtol_n)
    else
        # inner frequency integral
        integrand = ElectronDensityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), quad_n_ω, w; Σ, β, abstol=atol_n*det(bz.B)/det(bz.B)/nsyms(bz), reltol=rtol_n)
        IntegralSolver(integrand, bz, quad_n_k; abstol=atol_n*det(bz.B), reltol=rtol_n)
    end
    return ρ, det(bz.B), info
end

function density_interp(; cache_file_interp_density="cache-interp-density.jld2", kws...)
    (; atol_n, rtol_n, lims_μ, interptolratio, prec, cache_dir, nthreads) = merge(default, NamedTuple(kws))

    ρ, V, info_solver = density_solver(; kws..., atol_n=atol_n/interptolratio, rtol_n=rtol_n/interptolratio)
    info = (; info_solver..., atol_n, rtol_n, interptolratio, prec, lims_μ)
    id = string(info)

    lb, ub = map(prec, lims_μ)
    cache_path = joinpath(cache_dir, cache_file_interp_density)
    @info "Density interpolation" info...
    ρ_interp = cache_hchebinterp(lb, ub, atol_n*V, rtol_n, cache_path, id) do μ
        batchsolve(ρ, paramzip(; μ); nthreads=nthreads)
    end
    return ((; μ) -> ρ_interp(μ)), V, info
end

function density_batchsolve(; μ_series, cache_file_values_density="cache-values-density.jld2", kws...)
    (; prec, cache_dir, nthreads) = merge(default, NamedTuple(kws))

    ρ, V, info_solver = density_solver(; kws...)
    info = (; info_solver..., prec, μ=hash(μ_series))
    id = string(info)

    cache_path = joinpath(cache_dir, cache_file_values_density)
    @info "Density evaluation" info...
    data = cache_batchsolve(ρ, paramzip(; μ=prec.(ω_series)), cache_path, id, nthreads)
    return data, V, info
end

function findchempot(; cache_filroot_es_chempot="cache-roots-chempot.jld2", kws...)
    (; atol_n, rtol_n, root_n_μ, lims_μ, ν, nsp, prec, interp_μ, cache_dir) = merge(default, NamedTuple(kws))

    ρ, V, info_density = interp_μ ? density_interp(; kws...) : density_solver(; kws...)
    info = (; info_density..., atol_n, rtol_n, prec, lims_μ, ν, nsp, root_n_μ, interp_μ)
    id = string(info)

    u = prec(oneunit(eltype(lims_μ)))
    lb, ub = map(x -> prec(x/u), lims_μ)
    p = (u, ρ, V, prec(ν/nsp))
    cache_path = joinpath(cache_dir, cache_filroot_es_chempot)
    @info "Chemical potential finding" info...
    μ = cache_rootsolve(lb, ub, p, root_n_μ, atol_n, rtol_n, cache_path, id) do μ, (u, ρ, V, ν)
        oftype(ν, ρ(; μ=μ*u)/V)-ν
    end
    return u*μ, V, info
end
