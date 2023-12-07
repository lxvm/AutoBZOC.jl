using HChebInterp
using NonlinearSolve
using JLD2
using Printf


function interpolatedensity(; kws...)
    (; kalg) = merge(default, NamedTuple(kws))
    if kalg isa PTR || kalg isa AutoPTR
        return interpolatedensitywk(; kws...)
    else
        return interpolatedensitykw(; kws...)
    end
end

function interpolatedensitykw(; io=stdout, verb=true, cachepath=pwd(),
    nworkers=1, batchthreads=Threads.nthreads(), kws...)

    (; t, t′, Δ, ndim, natol, nrtol, μlims, bzkind, falg, kalg, tolratio, prec, gauge) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws...)
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)

    id = string((; t, t′, Δ, ndim, natol, nrtol, μlims, η, β, bzkind, falg, kalg, tolratio, prec, gauge))

    return jldopen(joinpath(cachepath, "cache-density-kw.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating density to add to cache" id

            Σ = EtaSelfEnergy(η)
            abstol = natol*det(bz.B)/tolratio
            reltol = nrtol/tolratio
            w = AutoBZCore.workspace_allocate(h, AutoBZCore.period(h), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
            integrand = ElectronDensityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg, w; Σ, β, abstol=abstol/det(bz.B)/nsyms(bz), reltol)
            solver = IntegralSolver(integrand, bz, kalg; abstol, reltol)

            cnt::Int = 0
            f = BatchFunction() do μ
                cnt += length(μ)
                dat = @timed batchsolve(solver, paramzip(; μ), typeof(float(det(bz.B))); nthreads=batchthreads)
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(μ) dat.time
                upreferred.(dat.value/det(bz.B))
            end

            fn[id] = stats = @timed hchebinterp(f, map(prec, μlims)...; atol=natol, rtol=nrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end


function interpolatedensitywk(; io=stdout, verb=true, cachepath=pwd(),
    nworkers=1, batchthreads=Threads.nthreads(), kws...)

    (; t, t′, Δ, ndim, natol, nrtol, μlims, bzkind, falg, kalg, tolratio, prec, gauge) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws...)
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)

    id = string((; t, t′, Δ, ndim, natol, nrtol, μlims, η, β, bzkind, falg, kalg, tolratio, prec, gauge))

    return jldopen(joinpath(cachepath, "cache-density-wk.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating density to add to cache" id

            Σ = EtaSelfEnergy(η)
            abstol = natol*det(bz.B)/tolratio
            reltol = nrtol/tolratio
            f = AutoBZ.parentseries(h)
            bandwidth_bound = sqrt(sum(norm(c)^2 for c in f.c) - norm(f.c[-CartesianIndex(f.o)])^2)
            w = AutoBZCore.workspace_allocate(h, AutoBZCore.period(h), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
            integrand = ElectronDensityIntegrand(bz, kalg, w; Σ, β, abstol=abstol/bandwidth_bound, reltol)
            solver = IntegralSolver(integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg; abstol, reltol)

            cnt::Int = 0
            f = BatchFunction() do μ
                cnt += length(μ)
                dat = @timed batchsolve(solver, paramzip(; μ), typeof(float(det(bz.B))); nthreads=batchthreads)
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(μ) dat.time
                upreferred.(dat.value/det(bz.B))
            end

            fn[id] = stats = @timed hchebinterp(f, map(prec, μlims)...; atol=natol, rtol=nrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end


function findchempot(; io=stdout, verb=true, kws...)

    (; nalg, μlims, ν, nsp, natol, nrtol, prec) = merge(default, NamedTuple(kws))

    n = interpolatedensity(; io, verb, kws...)

    verb && @info "Finding chemical potential"

    cnt::Int = 0
    u = prec(oneunit(eltype(μlims)))
    uμlims = map(x -> prec(x/u), μlims)
    prob = IntervalNonlinearProblem(uμlims, prec(ν/nsp)) do μ, ν
        cnt += 1
        return n(u*μ)-ν
    end
    stats = @timed solve(prob, nalg, abstol=natol, reltol=nrtol)

    verb && @printf io "Done finding chemical potential after %.3e s, %5i sample points\n" stats.time cnt

    return stats.value.u*u
end
