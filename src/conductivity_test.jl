using AutoBZ


struct IndexOrdering{T<:Base.Order.Ordering,I} <: Base.Order.Ordering
    o::T
    i::I
end

struct AuxArray{T,N,V<:AbstractArray{T,N}} <: AbstractArray{T,N}
    v::V
end

Base.Order.lt(o::IndexOrdering, a::Number, b::AuxArray) =
    Base.Order.lt(o.o, a, b.v[o.i])
Base.Order.lt(o::IndexOrdering, a::AuxArray, b::Number) =
    Base.Order.lt(o.o, a.v[o.i], b)
Base.Order.lt(o::IndexOrdering, a::AuxArray, b::AuxArray) =
    Base.Order.lt(o.o, a.v[o.i], b.v[o.i])
Base.Order.lt(o::IndexOrdering, a::AutoBZCore.IteratedIntegration.AuxQuadGK.Segment, b::AutoBZCore.IteratedIntegration.AuxQuadGK.Segment) =
    Base.Order.lt(o, a.E, b.E)

LinearAlgebra.norm(a::AuxArray) = AuxArray(map(norm, a.v))
Base.size(a::AuxArray) = size(a.v)
Base.axes(a::AuxArray) = axes(a.v)
Base.IndexStyle(::Type{AuxArray{T,N,V}}) where {T,N,V} = Base.IndexStyle(V)
Base.getindex(a::AuxArray, args...) = getindex(a.v, args...)
Base.zero(a::AuxArray) = AuxArray(zero(a.v))
Base.:+(a::AuxArray, b::AuxArray) = AuxArray(a.v + b.v)
Base.:-(a::AuxArray, b::AuxArray) = AuxArray(a.v - b.v)
Base.:*(a::Number, b::AuxArray) = AuxArray(a * b.v)
Base.:*(a::AuxArray, b::Number) = b * a
Base.:/(a::AuxArray, b::Number) = inv(b)*a
Base.:+(a::AuxArray) = AuxArray(+(a.v))
Base.:-(a::AuxArray) = AuxArray(-(a.v))

Base.isinf(a::AuxArray) = any(isinf, a.v)
Base.isnan(a::AuxArray) = any(isnan, a.v)
Base.isless(a::AuxArray, b::AuxArray) = all(splat(isless), zip(a.v, b.v))
Base.real(a::AuxArray) = AuxArray(real(a.v))
Base.imag(a::AuxArray) = AuxArray(imag(a.v))
Base.complex(a::AuxArray) = AuxArray(complex(a.v))

# first in the order of the array
AutoBZCore.IteratedIntegration.AuxQuadGK.eachorder(a::AuxArray) =
    map(i -> IndexOrdering(Base.Order.Reverse, i), eachindex(a.v))
AutoBZCore.symmetrize(f, bz, a::AuxArray) = AuxArray(map(x -> symmetrize(f, bz, x), a.v))
AutoBZCore.symmetrize(f, ::AutoBZCore.FullBZ, a::AuxArray) = a

AutoBZ._inv(a::AuxArray) = AuxArray(map(AutoBZ._inv, a.v))
AutoBZ.tr_inv(a::AuxArray) = AuxArray(map(AutoBZ.tr_inv, a.v))
AutoBZ.diag_inv(a::AuxArray) = AuxArray(map(AutoBZ.diag_inv, a.v))
AutoBZ.propagator_denominator(h::FourierValue, a::AuxArray) = AuxArray(map(M -> AutoBZ.propagator_denominator(h, M), a.v))
AutoBZ._evalM(Σs::AuxArray, ω, μ) = (AuxArray(map(Σ -> AutoBZ._evalM(Σ, ω, μ)[1], Σs)),)
AutoBZ.gloc_integrand(a::AuxArray) = AuxArray(map(AutoBZ.gloc_integrand, a.v))
AutoBZ.spectral_function(a::AuxArray) = AuxArray(map(AutoBZ.spectral_function, a.v))
AutoBZ.sigma_to_eta(a::AuxArray) = minimum(AutoBZ.sigma_to_eta, a.v)

struct VectorizedSelfEnergy{T,F} <: AutoBZ.AbstractSelfEnergy
    interpolants::T
    lb::F
    ub::F
end
(Σ::VectorizedSelfEnergy)(ω::Number) = AuxArray(Σ.interpolants(ω))
AutoBZ.lb(Σ::VectorizedSelfEnergy) = Σ.lb
AutoBZ.ub(Σ::VectorizedSelfEnergy) = Σ.ub

function series_selfenergy(; series_scattering, kws...)
    (; selfenergy, prec) = merge(default, NamedTuple(kws))
    Σ, info_selfenergy = selfenergy(; kws...)
    info = (; info_selfenergy..., prec, series_scattering)
    series_η = map(prec, series_scattering)
    series_Σ = VectorizedSelfEnergy(AutoBZ.lb(Σ), AutoBZ.ub(Σ)) do ω
        Σ0 = Σ(ω)
        map(η -> Σ0 - (im*η)*I, series_η)
    end
    return series_Σ, info
end

AutoBZ.transport_distribution_integrand_(vs::SVector, Aω::AuxArray) = AuxArray(map(A -> AutoBZ.transport_distribution_integrand_(vs, A), Aω.v))
AutoBZ.transport_distribution_integrand_(vs::SVector, Aω1::AuxArray, Aω2::AuxArray,) = AuxArray(map((A1, A2) -> AutoBZ.transport_distribution_integrand_(vs, A1, A2), Aω1.v, Aω2.v))

function test_transport_distribution_integrand(v::AutoBZ.FourierValue, callback::F, Mω₁, Mω₂, isdistinct) where {F}
    h, vs = v.s
    vh = FourierValue(v.x, h)
    callback()
    if isdistinct
        Aω₁ = AutoBZ.spectral_function(vh, Mω₁)
        Aω₂ = AutoBZ.spectral_function(vh, Mω₂)
        return AutoBZ.transport_distribution_integrand_(vs, Aω₁, Aω₂)
    else
        Aω = AutoBZ.spectral_function(vh, Mω₁)
        return AutoBZ.transport_distribution_integrand_(vs, Aω)
    end
end
function test_transport_fermi_integrand_inside(ω, callback::F; n, β, Ω, μ, Σ, hv_k) where {F}
    Γ = test_transport_distribution_integrand(hv_k, callback, AutoBZ.evalM2(; Σ, ω₁=ω, ω₂=ω+Ω, μ)...)
    return AutoBZ.transport_fermi_integrand_(ω, Γ, n, β, Ω)
end
function test_transport_fermi_integrand_inside(ω, callback::F, ::AutoBZ.CanonicalParameters; Σ, hv_k, kws...) where {F}
    Σ_smooth = if Σ isa VectorizedSelfEnergy
        VectorizedSelfEnergy(ω -> AuxArray(map(Σω -> Σω - (im*oneunit(ω))*I, Σ(ω))), AutoBZ.lb(Σ), AutoBZ.ub(Σ))
    elseif Σ isa AutoBZ.AbstractSelfEnergy
        MatrixSelfEnergy(ω -> Σ(ω) - (im*oneunit(ω))*I, AutoBZ.lb(Σ), AutoBZ.ub(Σ))
    elseif Σ isa AuxArray
        AuxArray(map(Σω -> Σω - (im*oneunit(ω))*I, Σ))
    else
        error("unexpected self energy")
    end
    return test_transport_fermi_integrand_inside(ω, callback; Σ=Σ_smooth, hv_k, n=0, β=inv(oneunit(ω)), Ω=zero(ω), μ=zero(ω))
end
const TestKCFrequencyInsideType = ParameterIntegrand{typeof(test_transport_fermi_integrand_inside)}

function AutoBZCore.init_solver_cacheval(f::TestKCFrequencyInsideType, dom, alg)
    return AutoBZ._init_solver_cacheval(f, dom, alg)
end

struct TestKCFrequencyIntegral{T}
    solver::T
end


function (kc::TestKCFrequencyIntegral)(hv_k, dom, Σ, n, β, Ω, μ)
    AutoBZ._check_selfenergy_limits(Σ, dom)
    solver = IntegralSolver(kc.solver.f, dom, kc.solver.alg, kc.solver.cacheval, kc.solver.kwargs)
    return AutoBZ.kinetic_coefficient_integrand(hv_k, solver, Σ, n, β, Ω, μ)
end

function (kc::TestKCFrequencyIntegral)(hv_k, dom; kws...)
    return kc(hv_k, AutoBZ.kc_inner_params(dom; kws...)...)
end
function (kc::TestKCFrequencyIntegral)(x, dom, ::AutoBZ.CanonicalParameters; Σ, kws...)
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

const TestKineticCoefficientIntegrandType = FourierIntegrand{<:TestKCFrequencyIntegral}

function AutoBZCore.init_solver_cacheval(f::TestKineticCoefficientIntegrandType, dom, alg)
    return AutoBZ._init_solver_cacheval(f, dom, alg)
end

function AutoBZCore.remake_integrand_cache(f::TestKineticCoefficientIntegrandType, dom, p, alg, cacheval, kwargs)
    # pre-evaluate the self energy when remaking the cache
    new_p = AutoBZ.canonize(AutoBZ.kc_inner_params, p)
    # Define default equispace grid stepping
    new_alg = AutoBZ.choose_autoptr_step(alg, AutoBZ.sigma_to_eta(p.Σ), f.w.series)
    return AutoBZ._remake_integrand_cache(f, dom, new_p, new_alg, cacheval, kwargs)
end
AutoBZ.SymRep(kc::TestKineticCoefficientIntegrandType) = AutoBZ.coord_to_rep(kc.w.series)


function TestKineticCoefficientIntegrand(lb_, ub_, alg::AutoBZCore.IntegralAlgorithm, w::AutoBZCore.FourierWorkspace{<:AutoBZ.AbstractVelocityInterp}, callback=nothing; Σ, kwargs...)
    solver_kws, kws = AutoBZ.nested_solver_kwargs(NamedTuple(kwargs))
    # put the frequency integral inside otherwise
    frequency_integrand = ParameterIntegrand(test_transport_fermi_integrand_inside, callback; Σ=Σ(zero(lb_)), hv_k=FourierValue(AutoBZ.period(w.series), w(AutoBZ.period(w.series))))
    dom = AutoBZCore.PuncturedInterval((lb_, ub_))
    frequency_solver = IntegralSolver(frequency_integrand, dom, alg; solver_kws...)
    int = TestKCFrequencyIntegral(frequency_solver)
    p = ParameterIntegrand(int, dom; Σ, kws...)
    nest = AutoBZ.make_fourier_nest(p, ParameterIntegrand(int), w)
    return nest === nothing ? FourierIntegrand(p, w) : FourierIntegrand(p, w, nest)
end

function conductivity_test_solver(; μ, bandwidth_bound, kws...)
    (; model, series_selfenergy, choose_kω_order, quad_σ_k, quad_σ_ω, atol_σ, rtol_σ, vcomp, gauge, coord, nworkers, callback) = merge(default, NamedTuple(kws))

    h, bz, info_model = model(; kws..., gauge=Wannier())
    Σ, info_selfenergy = series_selfenergy(; kws...)
    β = invtemp(; kws...)
    is_order_kω = choose_kω_order(quad_σ_k, quad_σ_ω)
    info = (; model=info_model, series_selfenergy=info_selfenergy, β, μ, vcomp, gauge, coord, quad_σ_ω, quad_σ_k, is_order_kω, atol_σ, rtol_σ, callback)

    hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
    w = AutoBZCore.workspace_allocate_vec(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(hv)) : nworkers))
    σ = if !is_order_kω
        @assert is_order_kω
    else
        integrand = TestKineticCoefficientIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), quad_σ_ω, w, callback; Σ, β, μ, n=0, abstol=atol_σ/det(bz.B)/nsyms(bz), reltol=rtol_σ)
        IntegralSolver(integrand, bz, quad_σ_k; abstol=atol_σ, reltol=rtol_σ)
    end
    return σ, info
end

function benchmark_conductivity_test(; Ω, cache_file_bench_cond_test="cache-bench-cond-test.jld2", kws...)
    (; prec, atol_σ, cache_dir) = merge(default, NamedTuple(kws))

    auxfun_cnt = AuxCounter(nothing)
    σ, info_solver = conductivity_test_solver(; kws..., atol_σ, callback=auxfun_cnt, bandwidth_bound=prec(Ω))
    info = (; info_solver..., Ω, prec)
    id = string(info)
    cache_path = joinpath(cache_dir, cache_file_bench_cond_test)

    @info "Conductivity benchmark" info...
    data = cache_benchmark(σ, (), (; Ω=prec(Ω)), cache_path, id; kws...)
    return data, info
end
