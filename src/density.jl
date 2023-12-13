using HChebInterp
using NonlinearSolve
using JLD2
using Printf

function solverdensity(; kws...)
    (; natol, nrtol, nfalg, nkalg, nworkers,) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws...)
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    Σ = EtaSelfEnergy(η)
    w = AutoBZCore.workspace_allocate(h, AutoBZCore.period(h), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
    abstol = natol*det(bz.B)
    reltol = nrtol

    if nkalg isa PTR || nkalg isa AutoPTR
        # inner BZ integral
        f = AutoBZ.parentseries(h)
        bandwidth_bound = sqrt(sum(norm(c)^2 for c in f.c) - norm(f.c[-CartesianIndex(f.o)])^2)
        integrand = ElectronDensityIntegrand(bz, nkalg, w; Σ, β, abstol=abstol/bandwidth_bound, reltol)
        return IntegralSolver(integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), nfalg; abstol, reltol)
    else
        # inner frequency integral
        integrand = ElectronDensityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), nfalg, w; Σ, β, abstol=abstol/det(bz.B)/nsyms(bz), reltol)
        return IntegralSolver(integrand, bz, nkalg; abstol, reltol)
    end
end

function interpolatedensity(; io=stdout, verb=true, cachepath=pwd(),
    batchthreads=Threads.nthreads(), kws...)

    (; t, t′, Δ, ndim, natol, nrtol, μlims, bzkind, nfalg, nkalg, tolratio, prec, gauge) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws...)
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)

    id = string((; t, t′, Δ, ndim, natol, nrtol, μlims, η, β, bzkind, nfalg, nkalg, tolratio, prec, gauge))

    return jldopen(joinpath(cachepath, "cache-density-interp.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating density to add to cache" id

            ρ = solverdensity(; kws..., natol=natol/tolratio, nrtol=nrtol/tolratio)
            cnt::Int = 0
            f = BatchFunction() do μ
                cnt += length(μ)
                dat = @timed batchsolve(ρ, paramzip(; μ), typeof(float(det(bz.B))); nthreads=batchthreads)
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(μ) dat.time
                dat.value
            end

            fn[id] = stats = @timed hchebinterp(f, map(prec, μlims)...; atol=natol*det(bz.B), rtol=nrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end


function findchempot(; io=stdout, verb=true, cachepath=pwd(), kws...)

    (; t, t′, Δ, ndim, natol, nrtol, nalg, μlims, bzkind, nfalg, nkalg, tolratio, ν, nsp, prec, gauge, interp) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws...)
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)

    id = string((; t, t′, Δ, η, β, ndim, natol, nrtol, interp, μlims, bzkind, nfalg, nkalg, tolratio, ν, nsp, prec, gauge))

    return jldopen(joinpath(cachepath, "cache-chempot.jld2"), "a+") do fn
        if !haskey(fn, id)
            ρ = if interp
                interpolatedensity(; io, verb, kws...)
            else
                solver = solverdensity(; kws...)
                μ -> solver(; μ)
            end
            verb && @info "Finding chemical potential to add to cache" id

            cnt::Int = 0
            u = prec(oneunit(eltype(μlims)))
            uμlims = map(x -> prec(x/u), μlims)
            prob = IntervalNonlinearProblem(uμlims, (prec(ν/nsp), det(bz.B))) do μ, (ν, V)
                cnt += 1
                return oftype(ν, ρ(μ*u)/V)-ν
            end
            stats = @timed(solve(prob, nalg, abstol=natol, reltol=nrtol))

            fn[id] = merge(stats, (; value=stats.value.u*u))

            verb && @printf io "Done finding chemical potential after %.3e s, %5i sample points\n" stats.time cnt
        end
        fn[id].value
    end

end
