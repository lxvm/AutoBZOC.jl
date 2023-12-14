using HChebInterp
using JLD2
using Printf
using FourierSeriesEvaluators: period


function solverconductivity(; μ, bandwidth_bound, kws...)
    (; σkalg, σfalg, σatol, σrtol, vcomp, gauge, coord, nworkers) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    shift!(h, μ)

    hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
    Σ = EtaSelfEnergy(η)
    abstol = σatol
    reltol = σrtol
    w = AutoBZCore.workspace_allocate_vec(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))

    if σkalg isa PTR || σkalg isa AutoPTR
        a, b = AutoBZ.fermi_window_limits(bandwidth_bound, β)
        len = b-a
        integrand = OpticalConductivityIntegrand(bz, σkalg, w; Σ, β, abstol=abstol/len, reltol)
        return IntegralSolver(integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), σfalg; abstol, reltol)
    else
        integrand = OpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), σfalg, w; Σ, β, abstol=abstol/det(bz.B)/nsyms(bz), reltol)
        return IntegralSolver(integrand, bz, σkalg; abstol, reltol)
    end
end

function interpolateconductivity(; io=stdout, verb=true, cachepath=pwd(),
    batchthreads=Threads.nthreads(), kws...)

    (; t, t′, Δ, ndim, Ωlims, σkalg, σfalg, σatol, σrtol, tolratio, vcomp, bzkind, prec, gauge, coord) = merge(default, NamedTuple(kws))

    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)

    id = string((; t, t′, Δ, ndim, η, β, μ, Ωlims, σkalg, σfalg, σatol, σrtol, tolratio, vcomp, bzkind, prec, gauge, coord))

    return jldopen(joinpath(cachepath, "cache-conductivity-interp.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating conductivity to add to cache" id

            solver = solverconductivity(; kws..., μ, bandwidth_bound=maximum(Ωlims), σatol=σatol/tolratio, σrtol=σrtol/tolratio)
            cnt::Int = 0
            f = BatchFunction() do Ω
                cnt += length(Ω)
                dat = @timed batchsolve(solver, paramzip(; Ω); nthreads=batchthreads)
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(Ω) dat.time
                dat.value
            end

            fn[id] = stats = @timed hchebinterp(f, map(prec, Ωlims)..., atol=σatol, rtol=σrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end

function batchsolveconductivity(; io=stdout, verb=true, cachepath=pwd(),
    batchthreads=Threads.nthreads(), kws...)

    (; t, t′, Δ, ndim, Ωseries, Tseries, σkalg, σfalg, σatol, σrtol, vcomp, bzkind, prec, gauge, coord) = merge(default, NamedTuple(kws))

    ηseries = [fermi_liquid_scattering(; kws..., T) for T in Tseries]
    βseries = [fermi_liquid_beta(; kws..., T) for T in Tseries]
    μseries = [findchempot(; io, verb, cachepath, kws..., T) for T in Tseries]

    id = string((; t, t′, Δ, ndim, Ω=hash(Ωseries), μ=hash(μseries), η=hash(ηseries), β=hash(βseries), σkalg, σfalg, σatol, σrtol, vcomp, bzkind, prec, gauge, coord))

    return jldopen(joinpath(cachepath, "cache-conductivity.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Solving conductivity to add to cache" id

            solver = solverconductivity(; kws..., μ=zero(eltype(μseries)), bandwidth_bound=maximum(Ωseries))
            pseries =  merge.(paramzip(; Ω=prec.(Ωseries)), permutedims(paramzip(; μ=μseries, β=βseries, Σ=map(EtaSelfEnergy, ηseries))))

            fn[id] = stats = @timed batchsolve(solver, pseries; nthreads=batchthreads)

            verb && @printf io "Done solving after %.3e s, %5i parameters\n" stats.time length(pseries)
        end
        return fn[id].value
    end

end


# just do the bz integral
function interpolateconductivityk(; ωlims, Ω, μoffset=zero(Ω),
    io=stdout, verb=true, cachepath=pwd(), kws...)

    (; t, t′, Δ, ndim, σkalg, σfalg, σatol, σrtol, tolratio, vcomp, bzkind, prec, gauge, coord, nworkers) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)
    shift!(h, μ)

    id = string((; t, t′, Δ, ndim, η, β, ωlims, Ω, μ, μoffset, σkalg, σfalg, σatol, σrtol, tolratio, vcomp, bzkind, prec, gauge, coord))

    return jldopen(joinpath(cachepath, "cache-conductivity-k.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating conductivity frequency integrand to add to cache" id

            hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
            Σ = EtaSelfEnergy(η)
            a, b = AutoBZ.fermi_window_limits(Ω, β)
            atol = σatol / (b-a) # compute absolute tolerance for frequency integrand
            abstol = atol/tolratio # tighter tolerance for integration than interpolation
            reltol = σrtol/tolratio
            w = AutoBZCore.workspace_allocate(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
            integrand = OpticalConductivityIntegrand(bz, σkalg, w; Σ, β, Ω=prec(Ω), μ=prec(μoffset), abstol, reltol)

            cnt::Int = 0
            f = BatchFunction() do ω
                cnt += length(ω)
                dat = @timed integrand.(ω, Ref(AutoBZCore.NullParameters()))
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(ω) dat.time
                dat.value
            end

            fn[id] = stats = @timed hchebinterp(f, map(prec, ωlims)...; atol, rtol=σrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end

# just do the frequency integral
function interpolateconductivityw(; Ω, μoffset=zero(Ω),
    io=stdout, verb=true, cachepath=pwd(), kws...)

    (; t, t′, Δ, ndim, σkalg, σfalg, σatol, σrtol, tolratio, vcomp, bzkind, prec, gauge, coord, nworkers) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)
    shift!(h, μ)

    id = string((; t, t′, Δ, ndim, η, β, μ, Ω, μoffset, σkalg, σfalg, σatol, σrtol, tolratio, vcomp, bzkind, prec, gauge, coord))

    return jldopen(joinpath(cachepath, "cache-conductivity-w.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating conductivity k integrand to add to cache" id

            hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
            Σ = EtaSelfEnergy(η)
            atol = σatol/det(bz.B)  # corresponding tolerance of k integrand
            abstol = atol/tolratio/nsyms(bz) # tighter tolerance for integration than interpolation
            reltol = σrtol/tolratio
            w = AutoBZCore.workspace_allocate(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
            integrand = OpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), σfalg, w; Σ, β, Ω=prec(Ω), μ=prec(μoffset), abstol, reltol)

            cnt::Int = 0
            f = BatchFunction() do k
                cnt += length(k)
                dat = @timed integrand.(k, Ref(AutoBZCore.NullParameters()))
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(k) dat.time
                dat.value
            end

            fn[id] = stats = @timed hchebinterp(f, map(zero, period(hv)), period(hv); atol, rtol=σrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end
