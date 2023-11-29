using HChebInterp

function interpolateconductivitykw(; t=-0.25u"eV", t′=0.05u"eV", Δ=0.0u"eV",
    T = 100.0u"K", T₀=300.0u"K", Z=0.5, ν=1.0, nsp=2, vcomp=Whole(), kalg=PTR(npt=100), falg=QuadGKJL(),
    Ωlims=(zero(t), oneunit(t)), atol=0.0u"eV^2*Å^-1", rtol=1e-4)

    return jldopen(joinpath(pwd(), "conductivitykwcache.jld2"), "a+") do fn
        id = "t$(t)_t′$(t′)_Δ($Δ)_T($T)_T₀$(T₀)_Z$(Z)_ν$(ν)_nsp$(nsp)_kalg$(kalg)_falg$(falg)_atol$(atol)_rtol($rtol)_Ωlims$(Ωlims)_vcomp$(vcomp)"
        if !haskey(fn, id)
            n = interpolatedensity(; t=t, t′=t′, Δ=Δ)
            μ = findchempot(n; ν=ν, nsp=nsp)
            h = t2gmodel(t=t, t′=t′, Δ=Δ, gauge=Wannier())
            shift!(h, μ)
            bz = load_bz(CubicSymIBZ(), one(SMatrix{3,3,Float64,9}) * u"Å")
            hv = GradientVelocityInterp(h, bz.A; coord=Cartesian(), vcomp=vcomp)
            η = uconvert(unit(t), T[1]^2*u"k_au"*pi/(Z*T₀))
            Σ = ConstScalarSelfEnergy(-im*η)
            β = 1/uconvert(unit(t), u"k_au"*T[1])
            integrand = OpticalConductivityIntegrand(falg, hv, Σ, β, abstol=atol/det(bz.B)/nsyms(bz)/10, reltol=rtol/10)
            solver = IntegralSolver(integrand, bz, kalg, abstol=atol, reltol=rtol)
            solver(Ωlims[1])
            f = BatchFunction(x -> batchsolve(solver, x, nthreads=Threads.nthreads()))
            fn[id] = hchebinterp(f, Ωlims..., atol=atol, rtol=rtol)
        end
        return fn[id]
    end

end

# todo: opposite order of integration
