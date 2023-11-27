using HChebInterp
using NonlinearSolve

function interpolatedensity(; t=-0.25u"eV", t′=0.05u"eV", Δ=0.0u"eV",
    T = [100.0u"K"], T₀=300.0u"K", Z=0.5,
    μlims=(-8t, 8t), atol=1e0u"Å^-3", rtol=1e-4)
    h = t2gmodel(t=t, t′=t′, Δ=Δ)
    bz = load_bz(CubicSymIBZ(), one(SMatrix{3,3,Float64,9}) * u"Å")
    falg = QuadGKJL()
    kalg = IAI()
    η = T[1]^2*u"k_au"*pi/(Z*T₀)
    Σ = ConstScalarSelfEnergy(-im*η)
    β = 1/uconvert(unit(t), u"k_au"*T[1])
    integrand = ElectronDensityIntegrand(falg, h, Σ, β, abstol=atol/det(bz.B)/nsyms(bz), reltol=rtol)
    solver = IntegralSolver(integrand, bz, kalg, abstol=atol, reltol=rtol)
    return hchebinterp(x -> upreferred(solver(μ=x)/det(bz.B)), μlims..., atol=10*atol/det(bz.B), rtol=rtol*10)
    # batchsolve(solver, [10.0u"eV"])
end


function findchempot(density; μlims=(-2.0u"eV", 2.0u"eV"), ν=2.0, nsp=2, alg=Falsi(), atol=1e-5, rtol=1e-5)
    u = oneunit(eltype(μlims))
    uμlims = sort(map(x -> x/u, μlims))
    prob = IntervalNonlinearProblem((x, ν) -> density(u*x)-ν, uμlims, ν/nsp)
    return solve(prob, alg, abstol=atol, reltol=rtol).u*u
end
