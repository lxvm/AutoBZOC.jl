using HChebInterp
using JLD2
using Printf
using FourierSeriesEvaluators: period

# we need to overload this for the interpolation
function HChebInterp._chebinterp(data::AbstractArray{<:AuxValue}, args...; kws...)
    # u = oneunit(eltype(data))
    # c = chebinterp(data/u, args...; kws...)
    # return ChebPoly(u*c.coefs, c.lb, c.ub)
    cval = HChebInterp._chebinterp(getproperty.(data, Ref(:val)), args...; kws...)
    caux = HChebInterp._chebinterp(getproperty.(data, Ref(:aux)), args...; kws...)
    return HChebInterp.ChebPoly(map(AuxValue, cval.coefs, caux.coefs), cval.lb, caux.ub)
end

function solverauxconductivity(; μ, bandwidth_bound, kws...)
    (; σkalg, σfalg, σatol, σrtol, σauxatol, σauxrtol, vcomp, gauge, coord, auxfun, nworkers) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    shift!(h, μ)

    hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
    Σ = EtaSelfEnergy(η)
    abstol = AuxValue(σatol, σauxatol)
    reltol = AuxValue(σrtol, σauxrtol)
    w = AutoBZCore.workspace_allocate(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))

    if σkalg isa PTR || σkalg isa AutoPTR
        a, b = AutoBZ.fermi_window_limits(bandwidth_bound, β)
        len = b-a
        integrand = AuxOpticalConductivityIntegrand(bz, σkalg, w, auxfun; Σ, β, abstol=abstol/len, reltol)
        return IntegralSolver(integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), σfalg; abstol, reltol)
    else
        integrand = AuxOpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), σfalg, w, auxfun; Σ, β, abstol=abstol/det(bz.B)/nsyms(bz), reltol)
        return IntegralSolver(integrand, bz, σkalg; abstol, reltol)
    end
end

function interpolateauxconductivity(; io=stdout, verb=true, cachepath=pwd(),
    batchthreads=Threads.nthreads(), kws...)

    (; t, t′, Δ, ndim, Ωlims, σkalg, σfalg, σatol, σrtol, σauxatol, σauxrtol, tolratio, vcomp, bzkind, prec, gauge, coord, auxfun) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)
    shift!(h, μ)

    id = string((; t, t′, Δ, ndim, η, β, μ, Ωlims, σkalg, σfalg, σatol, σrtol, σauxatol, σauxrtol, tolratio, vcomp, bzkind, prec, gauge, coord, auxfun))

    return jldopen(joinpath(cachepath, "cache-auxconductivity-interp.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating auxilary conductivity to add to cache" id

            solver = solverauxconductivity(; kws..., μ, bandwidth_bound=maximum(Ωlims), σatol=σatol/tolratio, σrtol=σrtol/tolratio)

            cnt::Int = 0
            f = BatchFunction() do Ω
                cnt += length(Ω)
                dat = @timed batchsolve(solver, paramzip(; Ω); nthreads=batchthreads)
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(Ω) dat.time
                dat.value
            end

            fn[id] = stats = @timed hchebinterp(f, map(prec, Ωlims)..., atol=σauxatol, rtol=σauxrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end

function batchsolveauxconductivity(; io=stdout, verb=true, cachepath=pwd(),
    batchthreads=Threads.nthreads(), kws...)

    (; t, t′, Δ, ndim, Ωseries, Tseries, σkalg, σfalg, σatol, σrtol, σauxatol, σauxrtol, vcomp, bzkind, prec, gauge, coord, auxfun) = merge(default, NamedTuple(kws))

    ηseries = [fermi_liquid_scattering(; kws..., T) for T in Tseries]
    βseries = [fermi_liquid_beta(; kws..., T) for T in Tseries]
    μseries = [findchempot(; io, verb, cachepath, kws..., T) for T in Tseries]

    id = string((; t, t′, Δ, ndim, Ω=hash(Ωseries), μ=hash(μseries), η=hash(ηseries), β=hash(βseries), σkalg, σfalg, σatol, σrtol, σauxatol, σauxrtol, vcomp, bzkind, prec, gauge, coord, auxfun))

    return jldopen(joinpath(cachepath, "cache-auxconductivity.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Solving auxiliary conductivity to add to cache" id

            solver = solverauxconductivity(; kws..., μ=zero(eltype(μseries)), bandwidth_bound=maximum(Ωseries))
            pseries =  merge.(paramzip(; Ω=prec.(Ωseries)), permutedims(paramzip(; μ=μseries, β=βseries, Σ=map(EtaSelfEnergy, ηseries))))

            fn[id] = stats = @timed batchsolve(solver, pseries; nthreads=batchthreads)

            verb && @printf io "Done solving after %.3e s, %5i parameters\n" stats.time length(pseries)
        end
        return fn[id].value
    end

end


# just do the bz integral
function interpolateauxconductivityk(; ωlims, Ω, μoffset=zero(Ω),
    io=stdout, verb=true, cachepath=pwd(), kws...)

    (; t, t′, Δ, ndim, σkalg, σfalg, σatol, σrtol, σauxatol, σauxrtol, tolratio, vcomp, bzkind, prec, gauge, coord, auxfun, nworkers) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)
    shift!(h, μ)

    id = string((; t, t′, Δ, ndim, η, β, ωlims, Ω, μ, μoffset, σkalg, σfalg, σatol, σrtol, σauxatol, σauxrtol, tolratio, vcomp, bzkind, prec, gauge, coord, auxfun))

    return jldopen(joinpath(cachepath, "cache-auxconductivity-k.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating auxilary conductivity frequency integrand to add to cache" id

            hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
            Σ = EtaSelfEnergy(η)
            a, b = AutoBZ.fermi_window_limits(Ω, β)
            atol = AuxValue(σatol, σauxatol) / (b-a) # compute absolute tolerance for frequency integrand
            abstol = atol/tolratio # tighter tolerance for integration than interpolation
            reltol = AuxValue(σrtol, σauxrtol)/tolratio
            w = AutoBZCore.workspace_allocate(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
            integrand = AuxOpticalConductivityIntegrand(bz, σkalg, w, auxfun; Σ, β, Ω=prec(Ω), μ=prec(μoffset), abstol, reltol)

            cnt::Int = 0
            f = BatchFunction() do ω
                cnt += length(ω)
                dat = @timed integrand.(ω, Ref(AutoBZCore.NullParameters()))
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(ω) dat.time
                dat.value
            end

            fn[id] = stats = @timed hchebinterp(f, map(prec, ωlims)...; atol, rtol=σauxrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end

# just do the frequency integral
function interpolateauxconductivityw(; Ω, μoffset=zero(Ω),
    io=stdout, verb=true, cachepath=pwd(), kws...)

    (; t, t′, Δ, ndim, σkalg, σfalg, σatol, σrtol, σauxatol, σauxrtol, tolratio, vcomp, bzkind, prec, gauge, coord, auxfun, nworkers) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)
    shift!(h, μ)

    id = string((; t, t′, Δ, ndim, η, β, μ, Ω, μoffset, σkalg, σfalg, σatol, σrtol, σauxatol, σauxrtol, tolratio, vcomp, bzkind, prec, gauge, coord, auxfun))

    return jldopen(joinpath(cachepath, "cache-auxconductivity-w.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating auxilary conductivity k integrand to add to cache" id

            hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
            Σ = EtaSelfEnergy(η)
            atol = AuxValue(σatol, σauxatol)/det(bz.B)  # corresponding tolerance of k integrand
            abstol = atol/tolratio/nsyms(bz) # tighter tolerance for integration than interpolation
            reltol = AuxValue(σrtol, σauxrtol)/tolratio
            w = AutoBZCore.workspace_allocate(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
            integrand = AuxOpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), σfalg, w, auxfun; Σ, β, Ω=prec(Ω), μ=prec(μoffset), abstol, reltol)

            cnt::Int = 0
            f = BatchFunction() do k
                cnt += length(k)
                dat = @timed integrand.(k, Ref(AutoBZCore.NullParameters()))
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(k) dat.time
                dat.value
            end

            fn[id] = stats = @timed hchebinterp(f, map(zero, period(hv)), period(hv); atol, rtol=σauxrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end
