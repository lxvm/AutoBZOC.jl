using HChebInterp
using NonlinearSolve
using JLD2
using Printf

function solverdensity(; kws...)
    (; model, self_energy, natol, nrtol, nfalg, nkalg, nworkers) = merge(default, NamedTuple(kws))

    h, bz, modelinfo = model(; kws...)
    Σ = self_energy(; kws...)
    β = inv_temp(; kws...)
    w = AutoBZCore.workspace_allocate_vec(h, AutoBZCore.period(h), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
    abstol = natol*det(bz.B)
    reltol = nrtol
    info = (; modelinfo, β, Σ, nfalg, nkalg, natol, nrtol)

    ρ = if nkalg isa PTR || nkalg isa AutoPTR
        # inner BZ integral
        f = AutoBZ.parentseries(h)
        bandwidth_bound = sqrt(sum(norm(c)^2 for c in f.c) - norm(f.c[-CartesianIndex(f.o)])^2)
        integrand = ElectronDensityIntegrand(bz, nkalg, w; Σ, β, abstol=abstol/bandwidth_bound, reltol)
        IntegralSolver(integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), nfalg; abstol, reltol)
    else
        # inner frequency integral
        integrand = ElectronDensityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), nfalg, w; Σ, β, abstol=abstol/det(bz.B)/nsyms(bz), reltol)
        IntegralSolver(integrand, bz, nkalg; abstol, reltol)
    end
    return ρ, det(bz.B), info
end

function interpolatedensity(; cachepath=pwd(), batchthreads=Threads.nthreads(), kws...)

    (; natol, nrtol, μlims, interptolratio, prec) = merge(default, NamedTuple(kws))

    ρ, V, solverinfo = solverdensity(; kws..., natol=natol/interptolratio, nrtol=nrtol/interptolratio)

    info = (; solverinfo..., natol, nrtol, interptolratio, prec, μlims)
    id = string(info)

    return jldopen(joinpath(cachepath, "cache-density-interp.jld2"), "a+") do fn
        if !haskey(fn, id)
            @info "Density interpolation started" cachepath info...

            cnt::Int = 0
            f = BatchFunction() do μ
                cnt += nbatch = length(μ)
                dat = @timed batchsolve(ρ, paramzip(; μ); nthreads=batchthreads)
                @debug "Density interpolation" batch_elapsed=dat.time batch_samples=nbatch
                dat.value
            end

            fn[id] = stats = @timed hchebinterp(f, map(prec, μlims)...; atol=natol*V, rtol=nrtol)
            @info "Density interpolation finished" elapsed=stats.time samples=cnt
        end
        ρ_interp = fn[id].value
        return ((; μ) -> ρ_interp(μ)), V, info
    end

end


function findchempot(; cachepath=pwd(), kws...)

    (; natol, nrtol, nalg, μlims, ν, nsp, prec, interp) = merge(default, NamedTuple(kws))

    ρ, V, densityinfo = interp ? interpolatedensity(; kws...) : solverdensity(; kws...)

    info = (; densityinfo..., natol, nrtol, prec, μlims, ν, nsp, nalg)
    id = string(info)

    return jldopen(joinpath(cachepath, "cache-chempot.jld2"), "a+") do fn
        if !haskey(fn, id)
            @info "Chemical potential started" cachepath info...

            cnt::Int = 0
            u = prec(oneunit(eltype(μlims)))
            uμlims = map(x -> prec(x/u), μlims)
            prob = IntervalNonlinearProblem(uμlims, (ρ, V, prec(ν/nsp))) do μ, (ρ, V, ν)
                cnt += 1
                return oftype(ν, ρ(; μ=μ*u)/V)-ν
            end
            stats = @timed(solve(prob, nalg, abstol=natol, reltol=nrtol))

            fn[id] = merge(stats, (; value=stats.value.u*u))

            @info "Chemical potential finished" elapsed=stats.time samples=cnt
        end
        fn[id].value, info
    end

end
