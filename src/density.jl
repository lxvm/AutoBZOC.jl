using HChebInterp
using NonlinearSolve
using JLD2
using Printf

function interpolatedensity(;
    io = stdout,
    verb = true,
    t = default.t,
    t′ = default.t′,
    Δ = default.Δ,
    atol = default.natol,
    rtol = default.nrtol,
    μlims = default.μlims,
    T = default.T,
    T₀ = default.T₀,
    Z = default.Z,
    bzkind = default.bzkind,
    falg = default.falg,
    kalg = default.kalg,
)

    return jldopen(joinpath(pwd(), "densitycache.jld2"), "a+") do fn
        id = "t$(t)_t′$(t′)_Δ($Δ)_T($T)_T₀$(T₀)_Z$(Z)_μlims$(μlims)_kalg$(kalg)_falg$(falg)_atol$(atol)_rtol($rtol)_bzkind$(bzkind)"
        if !haskey(fn, id)
            verb && @info "Interpolating density to add to cache" id
            ti = time()
            ni = 0
            function status(n)
                ni += n
                verb && @printf io "\t %.3e s elapsed, sampling %5i points\n" time()-ti n
            end
            h = t2gmodel(t=t, t′=t′, Δ=Δ)
            !iszero(Δ) && bzkind isa CubicSymIBZ && error("nonzero CFS breaks cubic symmetry in BZ")
            bz = load_bz(bzkind, one(SMatrix{3,3,Float64,9}) * u"Å")
            η = T[1]^2*u"k_au"*pi/(Z*T₀)
            Σ = ConstScalarSelfEnergy(-im*η)
            β = 1/uconvert(unit(t), u"k_au"*T[1])
            abstol = atol*det(bz.B)/10
            reltol = rtol/10
            integrand = ElectronDensityIntegrand(falg, h, Σ, β, abstol=abstol/det(bz.B)/nsyms(bz), reltol=reltol)
            solver = IntegralSolver(integrand, bz, kalg, abstol=abstol, reltol=reltol)
            f = BatchFunction() do x
                status(length(x))
                return upreferred.(batchsolve(solver, x, nthreads=Threads.nthreads())/det(bz.B))
            end
            fn[id] = hchebinterp(f, μlims..., atol=atol, rtol=rtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" time()-ti ni
        end
        return fn[id]
    end

end


function findchempot(;
    io = stdout,
    verb = true,
    t = default.t,
    t′ = default.t′,
    Δ = default.Δ,
    alg = default.nalg,
    atol = default.natol,
    rtol = default.nrtol,
    μlims = default.μlims,
    T = default.T,
    T₀ = default.T₀,
    Z = default.Z,
    ν = default.ν,
    nsp = default.nsp,
    falg = default.falg,
    kalg = default.kalg,
    bzkind = default.bzkind,
)
    n = interpolatedensity(; io=io, verb=verb, t=t, t′=t′, Δ=Δ, T=T, T₀=T₀, Z=Z, kalg=kalg, falg=falg, μlims=μlims, bzkind=bzkind, atol=atol, rtol=rtol)
    verb && @info "Finding chemical potential"
    ti = time()
    ni = 0
    u = oneunit(eltype(μlims))
    uμlims = sort(map(x -> x/u, μlims))
    prob = IntervalNonlinearProblem(uμlims, ν/nsp) do x, ν
        ni += 1
        return n(u*x)-ν
    end
    sol = solve(prob, alg, abstol=atol, reltol=rtol)
    verb && @printf io "Done finding chemical potential after %.3e s, %5i sample points\n" time()-ti ni
    return sol.u*u
end
