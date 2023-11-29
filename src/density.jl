using HChebInterp
using NonlinearSolve
using JLD2

function interpolatedensity(; t=-0.25u"eV", t′=0.05u"eV", Δ=0.0u"eV",
    T = 100.0u"K", T₀=300.0u"K", Z=0.5,
    μlims=sort((-8t, 8t)), atol=1e-2, rtol=1e-4)

    return jldopen(joinpath(pwd(), "densitycache.jld2"), "a+") do fn
        id = "t$(t)_t′$(t′)_Δ($Δ)_T($T)_T₀$(T₀)_Z$(Z)_μlims$(μlims)_atol$(atol)_rtol($rtol)"
        if !haskey(fn, id)
            h = t2gmodel(t=t, t′=t′, Δ=Δ)
            bz = load_bz(CubicSymIBZ(), one(SMatrix{3,3,Float64,9}) * u"Å")
            falg = QuadGKJL()
            kalg = IAI()
            η = T[1]^2*u"k_au"*pi/(Z*T₀)
            Σ = ConstScalarSelfEnergy(-im*η)
            β = 1/uconvert(unit(t), u"k_au"*T[1])
            abstol = atol*det(bz.B)/10
            integrand = ElectronDensityIntegrand(falg, h, Σ, β, abstol=abstol/det(bz.B)/nsyms(bz), reltol=rtol)
            solver = IntegralSolver(integrand, bz, kalg, abstol=abstol, reltol=rtol)
            f = BatchFunction(x -> upreferred.(batchsolve(solver, x, nthreads=Threads.nthreads())/det(bz.B)))
            fn[id] = hchebinterp(f, μlims..., atol=atol, rtol=rtol*10)
        end
        return fn[id]
    end

end


function findchempot(density; μlims=(-2.0u"eV", 2.0u"eV"), ν=1.0, nsp=2, alg=Falsi(), atol=1e-5, rtol=1e-5)
    u = oneunit(eltype(μlims))
    uμlims = sort(map(x -> x/u, μlims))
    prob = IntervalNonlinearProblem((x, ν) -> density(u*x)-ν, uμlims, ν/nsp)
    return solve(prob, alg, abstol=atol, reltol=rtol).u*u
end
