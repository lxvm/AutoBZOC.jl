using AutoBZ

# h, Σ, and μ are fixed parameters
# ω is an interpolation parameter
function trgloc_solver(; μ, kws...)
    (; model, selfenergy, quad_g_k, atol_g, rtol_g, nworkers) = merge(default, NamedTuple(kws))

    h, bz, info_model = model(; kws...)
    Σ, info_selfenergy = selfenergy(; kws...)
    info = (; model=info_model, selfenergy=info_selfenergy, μ, quad_g_k, atol_g, rtol_g)

    w = AutoBZCore.workspace_allocate_vec(h, AutoBZCore.period(h), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
    integrand = DOSIntegrand(w; Σ, μ)
    # integrand = TrGlocIntegrand(w; Σ, μ)
    g = IntegralSolver(integrand, bz, quad_g_k; abstol=atol_g, reltol=rtol_g)
    return g, info
end

function trgloc_interp(; cache_file_interp_gloc="cache-interp-gloc.jld2", kws...)
    (; lims_ω, atol_g, rtol_g, interp_tolratio, prec, cache_dir, nthreads) = merge(default, NamedTuple(kws))

    g, info_solver = trgloc_solver(; kws..., atol_g=atol_g/interp_tolratio, rtol_g=rtol_g/interp_tolratio)
    info = (; info_solver..., atol_g, rtol_g, interp_tolratio, prec, lims_ω)
    id = string(info)

    lb, ub = map(prec, lims_ω)
    cache_path = joinpath(cache_dir, cache_file_interp_gloc)
    @info "Green's function interpolation" info...
    g_interp = cache_hchebinterp(lb, ub, atol_g, rtol_g, cache_path, id) do ω
        batchsolve(g, paramzip(; ω); nthreads=nthreads)
    end
    return ((; ω) -> g_interp(ω)), info
end

function trgloc_batchsolve(; series_ω, cache_file_values_gloc="cache-values-gloc.jld2", kws...)
    (; prec, cache_dir, nthreads) = merge(default, NamedTuple(kws))

    g, info_solver = trgloc_solver(; kws...)
    info = (; info_solver..., prec, ω=hash(series_ω))
    id = string(info)

    cache_path = joinpath(cache_dir, cache_file_values_gloc)
    @info "Green's function evaluation" info...
    data = cache_batchsolve(g, paramzip(; ω=prec.(series_ω)), cache_path, id, nthreads)
    return data, info
end

function benchmark_trgloc(; ω, cache_file_bench_trgloc="cache-bench-trgloc.jld2", kws...)
    (; prec, quad_g_k, cache_dir) = merge(default, NamedTuple(kws))

    g, info_solver = trgloc_solver(; kws..., quad_g_k=EvalCounter(quad_g_k))
    info = (; info_solver..., ω, prec)
    id = string(info)
    cache_path = joinpath(cache_dir, cache_file_bench_trgloc)

    @info "Green's function benchmark" info...
    data = cache_benchmark((prec(ω)), (;), cache_path, id; kws...) do ω
        sol = AutoBZCore.solve_p(g, (; ω))
        gcnt[] = sol.numevals
        return (; sol=sol.u, numevals = sol.numevals)
    end
    return data, info
end
