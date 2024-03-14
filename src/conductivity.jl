using AutoBZ

# h, Σ, β, and μ are fixed parameters
# Ω is an interpolation parameter
# bandwidth_bound should be the largest Ω of interest
function conductivity_solver(; μ, bandwidth_bound, aux_inner_only=false, kws...)
    (; model, selfenergy, choose_kω_order, quad_σ_k, quad_σ_ω, atol_σ, rtol_σ, vcomp, gauge, coord, nworkers, auxfun) = merge(default, NamedTuple(kws))

    h, bz, info_model = model(; kws..., gauge=Wannier())
    Σ, info_selfenergy = selfenergy(; kws...)
    β = invtemp(; kws...)
    is_order_kω = choose_kω_order(quad_σ_k, quad_σ_ω)
    info = (; model=info_model, selfenergy=info_selfenergy, β, μ, vcomp, gauge, coord, quad_σ_ω, quad_σ_k, is_order_kω, atol_σ, rtol_σ, auxfun, aux_inner_only)

    hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
    w = AutoBZCore.workspace_allocate_vec(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(hv)) : nworkers))
    σ = if !is_order_kω
        a, b = AutoBZ.fermi_window_limits(bandwidth_bound, β)
        len = b-a
        integrand = if auxfun === nothing
            OpticalConductivityIntegrand(bz, quad_σ_k, w; Σ, β, μ, abstol=atol_σ/len, reltol=rtol_σ)
        else
            AuxOpticalConductivityIntegrand(bz, quad_σ_k, w, auxfun; Σ, β, μ, abstol=atol_σ/len, reltol=rtol_σ)
        end
        IntegralSolver(integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), quad_σ_ω; abstol=atol_σ, reltol=rtol_σ)
    else
        integrand = if auxfun === nothing
            _atol_σ = atol_σ
            OpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), quad_σ_ω, w; Σ, β, μ, abstol=atol_σ/det(bz.B)/nsyms(bz), reltol=rtol_σ)
        else
            _atol_σ = AuxValue(atol_σ.val, (aux_inner_only ? Inf : 1.0)*atol_σ.aux)
            AuxOpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), quad_σ_ω, w, auxfun; Σ, β, μ, abstol=atol_σ/det(bz.B)/nsyms(bz), reltol=rtol_σ)
        end
        IntegralSolver(integrand, bz, quad_σ_k; abstol=_atol_σ, reltol=rtol_σ)
    end
    return σ, info
end

function conductivity_interp(; cache_file_interp_cond="cache-interp-cond.jld2", kws...)
    (; lims_Ω, atol_σ, rtol_σ, interp_tolratio, prec, cache_dir, nthreads) = merge(default, NamedTuple(kws))

    σ, info_solver = conductivity_solver(; kws..., bandwidth_bound=maximum(lims_Ω), atol_σ=atol_σ/interp_tolratio, rtol_σ=rtol_σ/interp_tolratio)
    info = (; info_solver..., atol_σ, rtol_σ, interp_tolratio, prec, lims_Ω)
    id = string(info)

    lb, ub = map(prec, lims_Ω)
    cache_path = joinpath(cache_dir, cache_file_interp_cond)
    @info "Conductivity interpolation" info...
    σ_interp = cache_hchebinterp(lb, ub, atol_σ, rtol_σ, cache_path, id) do Ω
        batchsolve(σ, paramzip(; Ω); nthreads=nthreads)
    end
    return ((; Ω) -> σ_interp(Ω)), info
end

function conductivity_batchsolve(; series_Ω, cache_file_values_cond="cache-values-cond.jld2", kws...)
    (; prec, cache_dir, nthreads) = merge(default, NamedTuple(kws))

    σ, info_solver = conductivity_solver(; kws..., bandwidth_bound=maximum(series_Ω))
    info = (; info_solver..., prec, Ω=hash(series_Ω))
    id = string(info)

    cache_path = joinpath(cache_dir, cache_file_values_cond)
    @info "Conductivity evaluation" info...
    data = cache_batchsolve(σ, paramzip(; Ω=prec.(series_Ω)), cache_path, id, nthreads)
    return data, info
end

struct AuxCounter{F}
    auxfun::F
end
function (c::AuxCounter)(args...)
    gcnt[] += 1
    c.auxfun === nothing ? 1.0 : c.auxfun(args...)
end

function benchmark_conductivity(; Ω, cache_file_bench_cond="cache-bench-cond.jld2", kws...)
    (; model, prec, atol_σ, auxfun, cache_dir) = merge(default, NamedTuple(kws))

    _, bz, = model(; kws...)
    auxfun_cnt = AuxCounter(auxfun)
    atol_σ_aux = auxfun === nothing ? AuxValue(atol_σ, det(bz.B)) : atol_σ
    σ, info_solver = conductivity_solver(; kws..., atol_σ=atol_σ_aux, auxfun=auxfun_cnt, bandwidth_bound=prec(Ω))
    info = (; info_solver..., Ω, prec)
    id = string(info)
    cache_path = joinpath(cache_dir, cache_file_bench_cond)

    @info "Conductivity benchmark" info...
    data = cache_benchmark(σ, (), (; Ω=prec(Ω)), cache_path, id; kws...)
    return data, info
end
