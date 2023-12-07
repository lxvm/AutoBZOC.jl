using HChebInterp
using JLD2
using Printf
using StaticArrays
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

function interpolateauxconductivity(; kws...)
    (; kalg) = merge(default, NamedTuple(kws))
    if kalg isa PTR || kalg isa AutoPTR
        return interpolateauxconductivitywk(; kws...)
    else
        return interpolateauxconductivitykw(; kws...)
    end
end

function interpolateauxconductivitykw(; io=stdout, verb=true, cachepath=pwd(),
    nworkers=1, batchthreads=Threads.nthreads(), kws...)

    (; t, t′, Δ, ndim, Ωlims, kalg, falg, σatol, σrtol, σauxatol, σauxrtol, tolratio, vcomp, bzkind, prec, gauge, coord, auxfun) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)
    shift!(h, μ)

    id = string((; t, t′, Δ, ndim, η, β, μ, Ωlims, kalg, falg, σatol, σrtol, σauxatol, σauxrtol, tolratio, vcomp, bzkind, prec, gauge, coord, auxfun))

    return jldopen(joinpath(cachepath, "cache-auxconductivity-kw.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating auxilary conductivity to add to cache" id

            hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
            Σ = EtaSelfEnergy(η)
            abstol = AuxValue(σatol, σauxatol)/tolratio # tighter tolerance for integration than interpolation
            reltol = AuxValue(σrtol, σauxrtol)/tolratio # tighter tolerance for integration than interpolation
            w = AutoBZCore.workspace_allocate(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
            integrand = AuxOpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg, w, auxfun; Σ, β, abstol=abstol/det(bz.B)/nsyms(bz), reltol)
            solver = IntegralSolver(integrand, bz, kalg; abstol, reltol)

            # compute the type of the Auxiliary function
            hk, vk = hv(map(zero, period(hv)))
            Gk = AutoBZ.gloc_integrand(hk, -Σ(zero(μ)))
            T = typeof(auxfun(vk, Gk, Gk)*one(AutoBZ.lb(Σ))*det(bz.B))

            cnt::Int = 0
            f = BatchFunction() do Ω
                cnt += length(Ω)
                dat = @timed batchsolve(solver, paramzip(; Ω), AuxValue{SMatrix{ndim,ndim,typeof(complex(prec(σauxatol))),ndim^2},T}; nthreads=batchthreads)
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(Ω) dat.time
                dat.value
            end

            fn[id] = stats = @timed hchebinterp(f, map(prec, Ωlims)..., atol=σauxatol, rtol=σauxrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end


function interpolateauxconductivitywk(; io=stdout, verb=true, cachepath=pwd(),
    nworkers=1, batchthreads=Threads.nthreads(), kws...)

    (; t, t′, Δ, ndim, kalg, falg, Ωlims, σatol, σrtol, σauxatol, σauxrtol, tolratio, vcomp, bzkind, prec, gauge, coord, auxfun) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)
    shift!(h, μ)

    id = string((; t, t′, Δ, ndim, η, β, Ωlims, kalg, falg, σatol, σrtol, σauxatol, σauxrtol, tolratio, vcomp, bzkind, prec, gauge, coord, auxfun))

    return jldopen(joinpath(cachepath, "cache-auxconductivity-wk.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating auxilary conductivity to add to cache" id

            hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
            Σ = EtaSelfEnergy(η)
            abstol = AuxValue(σatol, σauxatol)/tolratio # tighter tolerance for integration than interpolation
            reltol = AuxValue(σrtol, σauxrtol)/tolratio
            a, b = AutoBZ.fermi_window_limits(maximum(Ωlims), β)
            len = b-a
            w = AutoBZCore.workspace_allocate(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
            integrand = AuxOpticalConductivityIntegrand(bz, kalg, w, auxfun; Σ, β, abstol=abstol/len, reltol)
            solver = IntegralSolver(integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg; abstol, reltol)

            # compute the type of the Auxiliary function
            hk, vk = hv(map(zero, period(hv)))
            Gk = AutoBZ.gloc_integrand(hk, -Σ(zero(μ)))
            T = typeof(auxfun(vk, Gk, Gk)*one(AutoBZ.lb(Σ))*det(bz.B))

            cnt::Int = 0
            f = BatchFunction() do Ω
                cnt += length(Ω)
                dat = @timed batchsolve(solver, paramzip(; Ω), AuxValue{SMatrix{ndim,ndim,typeof(complex(prec(σauxatol))),ndim^2},T}; nthreads=batchthreads)
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(Ω) dat.time
                dat.value
            end

            fn[id] = stats = @timed hchebinterp(f, map(prec, Ωlims)...; atol=σauxatol, rtol=σauxrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end

# just do the bz integral
function interpolateauxconductivityk(; ωlims, Ω, μoffset=zero(Ω),
    io=stdout, verb=true, cachepath=pwd(), nworkers=1, kws...)

    (; t, t′, Δ, ndim, kalg, falg, σatol, σrtol, σauxatol, σauxrtol, tolratio, vcomp, bzkind, prec, gauge, coord, auxfun) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)
    shift!(h, μ)

    id = string((; t, t′, Δ, ndim, η, β, ωlims, Ω, μ, μoffset, kalg, falg, σatol, σrtol, σauxatol, σauxrtol, tolratio, vcomp, bzkind, prec, gauge, coord, auxfun))

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
            integrand = AuxOpticalConductivityIntegrand(bz, kalg, w, auxfun; Σ, β, Ω=prec(Ω), μ=prec(μoffset), abstol, reltol)

            cnt::Int = 0
            f = BatchFunction() do ω
                cnt += length(ω)
                dat = @timed integrand.(ω, Ref(AutoBZ.CanonicalParameters()))
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
    io=stdout, verb=true, cachepath=pwd(), nworkers=1, kws...)

    (; t, t′, Δ, ndim, kalg, falg, σatol, σrtol, σauxatol, σauxrtol, tolratio, vcomp, bzkind, prec, gauge, coord, auxfun) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)
    shift!(h, μ)

    id = string((; t, t′, Δ, ndim, η, β, μ, Ω, μoffset, kalg, falg, σatol, σrtol, σauxatol, σauxrtol, tolratio, vcomp, bzkind, prec, gauge, coord, auxfun))

    return jldopen(joinpath(cachepath, "cache-auxconductivity-w.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating auxilary conductivity k integrand to add to cache" id

            hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
            Σ = EtaSelfEnergy(η)
            atol = AuxValue(σatol, σauxatol)/det(bz.B)  # corresponding tolerance of k integrand
            abstol = atol/tolratio/nsyms(bz) # tighter tolerance for integration than interpolation
            reltol = AuxValue(σrtol, σauxrtol)/tolratio
            w = AutoBZCore.workspace_allocate(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
            integrand = AuxOpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg, w, auxfun; Σ, β, Ω=prec(Ω), μ=prec(μoffset), abstol, reltol)

            cnt::Int = 0
            f = BatchFunction() do k
                cnt += length(k)
                dat = @timed integrand.(k, Ref(AutoBZ.CanonicalParameters()))
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(k) dat.time
                dat.value
            end

            fn[id] = stats = @timed hchebinterp(f, map(zero, period(hv)), period(hv); atol, rtol=σauxrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end
