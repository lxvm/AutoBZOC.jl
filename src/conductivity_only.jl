using AutoBZ

# function only_transport_distribution_integrand(v::AutoBZ.FourierValue, callback::F, Mω₁, Mω₂, isdistinct) where {F}
#     h, vs = v.s
#     vh = FourierValue(v.x, h)
#     callback()
#     if isdistinct
#         Aω₁ = AutoBZ.spectral_function(vh, Mω₁)
#         Aω₂ = AutoBZ.spectral_function(vh, Mω₂)
#         return AutoBZ.transport_distribution_integrand_(vs, Aω₁, Aω₂)
#     else
#         Aω = AutoBZ.spectral_function(vh, Mω₁)
#         return AutoBZ.transport_distribution_integrand_(vs, Aω)
#     end
# end
# function only_transport_fermi_integrand_inside(ω, callback::F; n, β, Ω, μ, Σ, hv_k) where {F}
#     Γ = only_transport_distribution_integrand(hv_k, callback, AutoBZ.evalM2(; Σ, ω₁=ω, ω₂=ω+Ω, μ)...)
#     return AutoBZ.transport_fermi_integrand_(ω, Γ, n, β, Ω)
# end
# function only_transport_fermi_integrand_inside(ω, callback::F, ::AutoBZ.CanonicalParameters; Σ, hv_k, kws...) where {F}
#     Σ_smooth = if Σ isa VectorizedSelfEnergy
#         VectorizedSelfEnergy(ω -> AuxArray(map(Σω -> Σω - (im*oneunit(ω))*I, Σ(ω))), AutoBZ.lb(Σ), AutoBZ.ub(Σ))
#     elseif Σ isa AutoBZ.AbstractSelfEnergy
#         MatrixSelfEnergy(ω -> Σ(ω) - (im*oneunit(ω))*I, AutoBZ.lb(Σ), AutoBZ.ub(Σ))
#     elseif Σ isa AuxArray
#         AuxArray(map(Σω -> Σω - (im*oneunit(ω))*I, Σ))
#     else
#         error("unexpected self energy")
#     end
#     return only_transport_fermi_integrand_inside(ω, callback; Σ=Σ_smooth, hv_k, n=0, β=inv(oneunit(ω)), Ω=zero(ω), μ=zero(ω))
# end
# const OnlyKCFrequencyInsideType = ParameterIntegrand{typeof(only_transport_fermi_integrand_inside)}

# function AutoBZCore.init_solver_cacheval(f::OnlyKCFrequencyInsideType, dom, alg)
#     return AutoBZ._init_solver_cacheval(f, dom, alg)
# end

struct OnlyKCFrequencyIntegral{T}
    solver::T
end


function (kc::OnlyKCFrequencyIntegral)(hv_k, dom, Σ, n, β, Ω, μ)
    AutoBZ._check_selfenergy_limits(Σ, dom)
    solver = IntegralSolver(kc.solver.f, dom, kc.solver.alg, kc.solver.cacheval, kc.solver.kwargs)
    return getval(AutoBZ.kinetic_coefficient_integrand(hv_k, solver, Σ, n, β, Ω, μ))
end

function (kc::OnlyKCFrequencyIntegral)(hv_k, dom; kws...)
    return kc(hv_k, AutoBZ.kc_inner_params(dom; kws...)...)
end
function (kc::OnlyKCFrequencyIntegral)(x, dom, ::AutoBZ.CanonicalParameters; Σ, kws...)
    el = real(AutoBZ._eltype(x.s[1]))
    Σ_smooth = if Σ isa VectorizedSelfEnergy
        VectorizedSelfEnergy(ω -> AuxArray(map(Σω -> Σω - (im*oneunit(ω))*I, Σ(ω).v)), AutoBZ.lb(Σ), AutoBZ.ub(Σ))
    elseif Σ isa AutoBZ.AbstractSelfEnergy
        MatrixSelfEnergy(ω -> Σ(ω) - (im*oneunit(ω))*I, AutoBZ.lb(Σ), AutoBZ.ub(Σ))
    elseif Σ isa AuxArray
        AuxArray(map(Σω -> Σω - (im*oneunit(ω))*I, Σ.v))
    else
        error("unexpected self energy")
    end
    return kc(x, dom; n=0, β=inv(oneunit(el)), Ω=zero(el), Σ=Σ_smooth)
end

const OnlyKineticCoefficientIntegrandType = FourierIntegrand{<:OnlyKCFrequencyIntegral}

function AutoBZCore.init_solver_cacheval(f::OnlyKineticCoefficientIntegrandType, dom, alg)
    return AutoBZ._init_solver_cacheval(f, dom, alg)
end

function AutoBZCore.remake_integrand_cache(f::OnlyKineticCoefficientIntegrandType, dom, p, alg, cacheval, kwargs)
    # pre-evaluate the self energy when remaking the cache
    new_p = AutoBZ.canonize(AutoBZ.kc_inner_params, p)
    # Define default equispace grid stepping
    new_alg = AutoBZ.choose_autoptr_step(alg, AutoBZ.sigma_to_eta(p.Σ), f.w.series)
    return AutoBZ._remake_integrand_cache(f, dom, new_p, new_alg, cacheval, kwargs)
end

AutoBZ.SymRep(kc::OnlyKineticCoefficientIntegrandType) = AutoBZ.coord_to_rep(kc.w.series)

function OnlyKineticCoefficientIntegrand(lb_, ub_, alg::AutoBZCore.IntegralAlgorithm, w::AutoBZCore.FourierWorkspace{<:AutoBZ.AbstractVelocityInterp}, callback=nothing; Σ, kwargs...)
    solver_kws, kws = AutoBZ.nested_solver_kwargs(NamedTuple(kwargs))
    # put the frequency integral inside otherwise
    frequency_integrand = ParameterIntegrand(AutoBZ.aux_transport_fermi_integrand_inside, callback; Σ=Σ(zero(lb_)), hv_k=FourierValue(AutoBZ.period(w.series), w(AutoBZ.period(w.series))))
    dom = AutoBZCore.PuncturedInterval((lb_, ub_))
    frequency_solver = IntegralSolver(frequency_integrand, dom, alg; solver_kws...)
    int = OnlyKCFrequencyIntegral(frequency_solver)
    p = ParameterIntegrand(int, dom; Σ, kws...)
    nest = AutoBZ.make_fourier_nest(p, ParameterIntegrand(int), w)
    return nest === nothing ? FourierIntegrand(p, w) : FourierIntegrand(p, w, nest)
end

function conductivity_only_solver(; μ, bandwidth_bound, kws...)
    (; model, selfenergy, choose_kω_order, quad_σ_k, quad_σ_ω, atol_σ, rtol_σ, vcomp, gauge, coord, nworkers, callback) = merge(default, NamedTuple(kws))

    h, bz, info_model = model(; kws..., gauge=Wannier())
    Σ, info_selfenergy = selfenergy(; kws...)
    β = invtemp(; kws...)
    is_order_kω = choose_kω_order(quad_σ_k, quad_σ_ω)
    info = (; model=info_model, auxinneronly=true, selfenergy=info_selfenergy, β, μ, vcomp, gauge, coord, quad_σ_ω, quad_σ_k, is_order_kω, atol_σ, rtol_σ, callback)

    hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
    w = AutoBZCore.workspace_allocate_vec(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(hv)) : nworkers))
    σ = if !is_order_kω
        @assert is_order_kω
    else
        integrand = OnlyKineticCoefficientIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), quad_σ_ω, w, callback; Σ, β, μ, n=0, abstol=atol_σ/det(bz.B)/nsyms(bz), reltol=rtol_σ)
        IntegralSolver(integrand, bz, quad_σ_k; abstol=getval(atol_σ), reltol=rtol_σ)
    end
    return σ, info
end

function benchmark_conductivity_only(; Ω, cache_file_bench_cond_only="cache-bench-cond-only.jld2", kws...)
    (; prec, atol_σ, auxfun, cache_dir) = merge(default, NamedTuple(kws))

    auxfun_cnt = AuxCounter(auxfun)
    atol_σ_aux = auxfun === nothing ? AuxValue(atol_σ, det(bz.B)) : atol_σ
    σ, info_solver = conductivity_only_solver(; kws..., atol_σ=atol_σ_aux, callback=auxfun_cnt, bandwidth_bound=prec(Ω))
    info = (; info_solver..., Ω, prec)
    id = string(info)
    cache_path = joinpath(cache_dir, cache_file_bench_cond_only)

    @info "Conductivity benchmark" info...
    data = cache_benchmark(σ, (), (; Ω=prec(Ω)), cache_path, id; kws...)
    return data, info
end
