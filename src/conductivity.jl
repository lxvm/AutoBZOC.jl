using HChebInterp
using JLD2
using Printf
using StaticArrays

function interpolateconductivitykw(;
    vcomp = Whole(),
    io = stdout,
    verb = true,
    t = default.t,
    t′ = default.t′,
    Δ = default.Δ,
    nalg = default.nalg,
    natol = default.natol,
    nrtol = default.nrtol,
    μlims = default.μlims,
    T = default.T,
    T₀ = default.T₀,
    Z = default.Z,
    Ωlims = default.Ωlims,
    atol = default.σatol,
    rtol = default.σrtol,
    ν = default.ν,
    nsp = default.nsp,
    falg = default.falg,
    kalg = default.kalg,
    tolratio = default.tolratio,
    bzkind = default.bzkind,
    prec = default.prec,
)

    return jldopen(joinpath(pwd(), "conductivitykwcache.jld2"), "a+") do fn
        id = "t$(t)_t′$(t′)_Δ($Δ)_T($T)_T₀$(T₀)_Z$(Z)_ν$(ν)_nsp$(nsp)_μlims$(μlims)_kalg$(kalg)_falg$(falg)_atol$(atol)_rtol($rtol)_tolratio$(tolratio)_Ωlims$(Ωlims)_vcomp$(vcomp)_bzkind$(bzkind)_natol$(natol)_nrtol$(nrtol)_nalg$(nalg)_prec$(prec)"
        if !haskey(fn, id)
            μ = findchempot(; io=io, verb=verb, t=t, t′=t′, Δ=Δ, T=T, T₀=T₀, Z=Z, μlims=μlims, ν=ν, nsp=nsp, alg=nalg, kalg=kalg, falg=falg, atol=natol, rtol=nrtol, bzkind=bzkind, tolratio=tolratio, prec=prec)
            verb && @info "Interpolating conductivity to add to cache" id
            ti = time()
            ni = 0
            function status(n, lb, ub)
                ni += n
                verb && @printf io "\t %.3e s elapsed, sampling %5i points in (%.3e, %.3e)\n" time()-ti n ustrip(lb) ustrip(ub)
            end
            h = t2gmodel(t=prec(t), t′=prec(t′), Δ=prec(Δ), gauge=Wannier())
            shift!(h, μ)
            bz = load_bz(bzkind, one(SMatrix{3,3,prec,9}) * u"Å")
            hv = GradientVelocityInterp(h, bz.A; coord=Cartesian(), vcomp=vcomp)
            η = prec(uconvert(unit(t), T[1]^2*u"k_au"*pi/(Z*T₀)))
            Σ = ConstScalarSelfEnergy(-im*η)
            β = prec(1/uconvert(unit(t), u"k_au"*T[1]))
            abstol = atol/tolratio # tighter tolerance for integration than interpolation
            reltol = rtol/tolratio # tighter tolerance for integration than interpolation
            integrand = OpticalConductivityIntegrand(falg, hv, Σ, β, abstol=abstol/det(bz.B)/nsyms(bz), reltol=reltol)
            solver = IntegralSolver(integrand, bz, kalg, abstol=abstol, reltol=reltol)
            f = BatchFunction() do x
                status(length(x), extrema(x)...)
                batchsolve(solver, x, SMatrix{3,3,typeof(complex(prec(atol))),9}, nthreads=Threads.nthreads())
            end
            fn[id] = hchebinterp(f, map(prec, Ωlims)..., atol=atol, rtol=rtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" time()-ti ni
        end
        return fn[id]
    end

end

# todo: opposite order of integration

function interpolateconductivitywk(;
    vcomp = Whole(),
    io = stdout,
    verb = true,
    t = default.t,
    t′ = default.t′,
    Δ = default.Δ,
    nalg = default.nalg,
    natol = default.natol,
    nrtol = default.nrtol,
    μlims = default.μlims,
    T = default.T,
    T₀ = default.T₀,
    Z = default.Z,
    Ωlims = default.Ωlims,
    atol = default.σatol,
    rtol = default.σrtol,
    ν = default.ν,
    nsp = default.nsp,
    falg = default.falg,
    kalg = default.kalg,
    tolratio = default.tolratio,
    bzkind = default.bzkind,
    prec = default.prec,
)

    return jldopen(joinpath(pwd(), "conductivitywkcache.jld2"), "a+") do fn
        id = "t$(t)_t′$(t′)_Δ($Δ)_T($T)_T₀$(T₀)_Z$(Z)_ν$(ν)_nsp$(nsp)_μlims$(μlims)_kalg$(kalg)_falg$(falg)_atol$(atol)_rtol($rtol)_tolratio$(tolratio)_Ωlims$(Ωlims)_vcomp$(vcomp)_bzkind$(bzkind)_natol$(natol)_nrtol$(nrtol)_nalg$(nalg)_prec$(prec)"
        if !haskey(fn, id)
            μ = findchempot(; io=io, verb=verb, t=t, t′=t′, Δ=Δ, T=T, T₀=T₀, Z=Z, μlims=μlims, ν=ν, nsp=nsp, alg=nalg, kalg=kalg, falg=falg, atol=natol, rtol=nrtol, bzkind=bzkind, tolratio=tolratio, prec=prec)
            verb && @info "Interpolating conductivity to add to cache" id
            ti = time()
            ni = 0
            function status(n)
                ni += n
                verb && @printf io "\t %.3e s elapsed, sampling %5i points\n" time()-ti n
            end
            h = t2gmodel(t=prec(t), t′=prec(t′), Δ=prec(Δ), gauge=Wannier())
            shift!(h, μ)
            bz = load_bz(CubicSymIBZ(), one(SMatrix{3,3,prec,9}) * u"Å")
            hv = GradientVelocityInterp(h, bz.A; coord=Cartesian(), vcomp=vcomp)
            η = prec(uconvert(unit(t), T[1]^2*u"k_au"*pi/(Z*T₀)))
            Σ = ConstScalarSelfEnergy(-im*η)
            β = prec(1/uconvert(unit(t), u"k_au"*T[1]))
            abstol = atol/tolratio # tighter tolerance for integration than interpolation
            reltol = rtol/tolratio
            integrand = OpticalConductivityIntegrand(bz, kalg, hv, Σ, β, abstol=abstol/det(bz.B)/nsyms(bz), reltol=reltol)
            solver = IntegralSolver(integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg, abstol=abstol, reltol=reltol)
            f = BatchFunction() do x
                status(length(x))
                batchsolve(solver, x, nthreads=Threads.nthreads())
            end
            fn[id] = hchebinterp(f, map(prec, Ωlims)..., atol=atol, rtol=rtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" time()-ti ni
        end
        return fn[id]
    end

end

# just do the bz integral
function interpolateconductivityk(;
    ωlims,
    vcomp = Inter(),
    Ω = default.Ωinter,
    μoffset = zero(Ω),
    io = stdout,
    verb = true,
    t = default.t,
    t′ = default.t′,
    Δ = default.Δ,
    nalg = default.nalg,
    natol = default.natol,
    nrtol = default.nrtol,
    μlims = default.μlims,
    T = default.T,
    T₀ = default.T₀,
    Z = default.Z,
    atol = default.σatol,
    rtol = default.σrtol,
    ν = default.ν,
    nsp = default.nsp,
    falg = default.falg,
    kalg = default.kalg,
    tolratio = default.tolratio,
    bzkind = default.bzkind,
    prec = default.prec,
)

    return jldopen(joinpath(pwd(), "conductivitykcache.jld2"), "a+") do fn
        id = "t$(t)_t′$(t′)_Δ($Δ)_Ω$(Ω)_T($T)_T₀$(T₀)_Z$(Z)_ν$(ν)_nsp$(nsp)_μlims$(μlims)_kalg$(kalg)_falg$(falg)_atol$(atol)_rtol($rtol)_tolratio$(tolratio)_ωlims$(ωlims)_vcomp$(vcomp)_bzkind$(bzkind)_natol$(natol)_nrtol$(nrtol)_nalg$(nalg)_μoffset$(μoffset)_prec$(prec)"
        if !haskey(fn, id)
            μ = findchempot(; io=io, verb=verb, t=t, t′=t′, Δ=Δ, T=T, T₀=T₀, Z=Z, μlims=μlims, ν=ν, nsp=nsp, alg=nalg, kalg=kalg, falg=falg, atol=natol, rtol=nrtol, bzkind=bzkind, tolratio=tolratio, prec=prec)
            verb && @info "Interpolating conductivity frequency integrand to add to cache" id
            ti = time()
            ni = 0
            function status(n)
                ni += n
                verb && @printf io "\t %.3e s elapsed, sampling %5i points\n" time()-ti n
            end
            h = t2gmodel(t=prec(t), t′=prec(t′), Δ=prec(Δ), gauge=Wannier())
            shift!(h, μ)
            bz = load_bz(CubicSymIBZ(), one(SMatrix{3,3,prec,9}) * u"Å")
            hv = GradientVelocityInterp(h, bz.A; coord=Cartesian(), vcomp=vcomp)
            w = AutoBZCore.workspace_allocate(hv, AutoBZCore.period(hv), (1, 1, Threads.nthreads()))
            η = prec(uconvert(unit(t), T[1]^2*u"k_au"*pi/(Z*T₀)))
            Σ = ConstScalarSelfEnergy(-im*η)
            β = prec(1/uconvert(unit(t), u"k_au"*T[1]))
            atol = atol / (Ω + 2/β) # compute absolute tolerance for frequency integrand
            abstol = atol/tolratio # tighter tolerance for integration than interpolation
            reltol = rtol/tolratio
            integrand = OpticalConductivityIntegrand(bz, kalg, w, Σ, β, prec(Ω),prec(μoffset), abstol=abstol/det(bz.B)/nsyms(bz), reltol=reltol)
            f = BatchFunction() do x
                status(length(x))
                integrand.(x, Ref(AutoBZCore.NullParameters()))
            end
            fn[id] = hchebinterp(f, map(prec, ωlims)..., atol=atol, rtol=rtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" time()-ti ni
        end
        return fn[id]
    end

end

# just do the frequency integral
function interpolateconductivityw(;
    vcomp = Inter(),
    Ω = default.Ωinter,
    io = stdout,
    verb = true,
    t = default.t,
    t′ = default.t′,
    Δ = default.Δ,
    nalg = default.nalg,
    natol = default.natol,
    nrtol = default.nrtol,
    μlims = default.μlims,
    T = default.T,
    T₀ = default.T₀,
    Z = default.Z,
    Ωlims = default.Ωlims,
    atol = default.σatol,
    rtol = default.σrtol,
    ν = default.ν,
    nsp = default.nsp,
    falg = default.falg,
    kalg = default.kalg,
    tolratio = default.tolratio,
    bzkind = default.bzkind,
    prec = default.prec,
)
#  t=-0.25u"eV", t′=0.05u"eV", Δ=0.0u"eV", Ω=zero(t),
#     T = 100.0u"K", T₀=300.0u"K", Z=0.5, ν=1.0, nsp=2, vcomp=Whole(), kalg=PTR(npt=100), falg=QuadGKJL(),
#     μlims=sort((-8t, 8t)), klims=(zeros(t), ones(t)), atol=0.0u"eV^2*Å^-1", rtol=1e-4,
#     bzkind=CubicSymIBZ(), nalg=Falsi(), natol=1e-5, nrtol=1e-5, io=stdout, verb=true)

    return jldopen(joinpath(pwd(), "conductivitywcache.jld2"), "a+") do fn
        id = "t$(t)_t′$(t′)_Δ($Δ)_Ω$(Ω)_T($T)_T₀$(T₀)_Z$(Z)_ν$(ν)_nsp$(nsp)_μlims$(μlims)_kalg$(kalg)_falg$(falg)_atol$(atol)_rtol($rtol)_tolratio$(tolratio)_klims$(klims)_vcomp$(vcomp)_bzkind$(bzkind)_natol$(natol)_nrtol$(nrtol)_nalg$(nalg)_prec$(prec)"
        if !haskey(fn, id)
            μ = findchempot(; io=io, verb=verb, t=t, t′=t′, Δ=Δ, T=T, T₀=T₀, Z=Z, μlims=μlims, ν=ν, nsp=nsp, alg=nalg, kalg=kalg, falg=falg, atol=natol, rtol=nrtol, bzkind=bzkind, tolratio=tolratio, prec=prec)
            verb && @info "Interpolating conductivity k integrand to add to cache" id
            ti = time()
            ni = 0
            function status(n)
                ni += n
                verb && @printf io "\t %.3e s elapsed, sampling %5i points\n" time()-ti n
            end
            h = t2gmodel(t=prec(t), t′=prec(t′), Δ=prec(Δ), gauge=Wannier())
            shift!(h, μ)
            bz = load_bz(CubicSymIBZ(), one(SMatrix{3,3,prec,9}) * u"Å")
            hv = GradientVelocityInterp(h, bz.A; coord=Cartesian(), vcomp=vcomp)
            η = prec(uconvert(unit(t), T[1]^2*u"k_au"*pi/(Z*T₀)))
            Σ = ConstScalarSelfEnergy(-im*η)
            β = prec(1/uconvert(unit(t), u"k_au"*T[1]))
            abstol = atol/tolratio # tighter tolerance for integration than interpolation
            reltol = rtol/tolratio
            integrand = OpticalConductivityIntegrand(bz, kalg, hv, Σ, β, abstol=abstol/det(bz.B)/nsyms(bz), reltol=reltol)
            solver = IntegralSolver(integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg, abstol=abstol, reltol=reltol)
            f = BatchFunction() do x
                status(length(x))
                batchsolve(solver, x, nthreads=Threads.nthreads())
            end
            fn[id] = hchebinterp(f, map(prec, Ωlims)..., atol=atol, rtol=rtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" time()-ti ni
        end
        return fn[id]
    end

end