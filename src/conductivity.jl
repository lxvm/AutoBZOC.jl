using HChebInterp
using JLD2
using Printf
using StaticArrays
using FourierSeriesEvaluators: period

function interpolateconductivity(; kws...)
    (; kalg) = merge(default, NamedTuple(kws))
    if kalg isa PTR || kalg isa AutoPTR
        return interpolateconductivitywk(; kws...)
    else
        return interpolateconductivitykw(; kws...)
    end
end

function interpolateconductivitykw(; io=stdout, verb=true, cachepath=pwd(),
    nworkers=1, batchthreads=Threads.nthreads(), kws...)

    (; t, t′, Δ, ndim, Ωlims, kalg, falg, σatol, σrtol, tolratio, vcomp, bzkind, prec, gauge, coord) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)
    shift!(h, μ)

    id = string((; t, t′, Δ, ndim, η, β, μ, Ωlims, kalg, falg, σatol, σrtol, tolratio, vcomp, bzkind, prec, gauge, coord))

    return jldopen(joinpath(cachepath, "cache-conductivity-kw.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating conductivity to add to cache" id

            hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
            Σ = EtaSelfEnergy(η)
            abstol = σatol/tolratio # tighter tolerance for integration than interpolation
            reltol = σrtol/tolratio # tighter tolerance for integration than interpolation
            w = AutoBZCore.workspace_allocate(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
            integrand = OpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg, w; Σ, β, abstol=abstol/det(bz.B)/nsyms(bz), reltol)
            solver = IntegralSolver(integrand, bz, kalg; abstol, reltol)

            cnt::Int = 0
            f = BatchFunction() do Ω
                cnt += length(Ω)
                dat = @timed batchsolve(solver, paramzip(; Ω), SMatrix{ndim,ndim,typeof(complex(prec(σatol))),ndim^2}; nthreads=batchthreads)
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(Ω) dat.time
                dat.value
            end

            fn[id] = stats = @timed hchebinterp(f, map(prec, Ωlims)..., atol=σatol, rtol=σrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end


function interpolateconductivitywk(; io=stdout, verb=true, cachepath=pwd(),
    nworkers=1, batchthreads=Threads.nthreads(), kws...)

    (; t, t′, Δ, ndim, kalg, falg, Ωlims, σatol, σrtol, tolratio, vcomp, bzkind, prec, gauge, coord) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)
    shift!(h, μ)

    id = string((; t, t′, Δ, ndim, η, β, Ωlims, kalg, falg, σatol, σrtol, tolratio, vcomp, bzkind, prec, gauge, coord))

    return jldopen(joinpath(cachepath, "cache-conductivity-wk.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating conductivity to add to cache" id

            hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
            Σ = EtaSelfEnergy(η)
            abstol = σatol/tolratio # tighter tolerance for integration than interpolation
            reltol = σrtol/tolratio
            a, b = AutoBZ.fermi_window_limits(maximum(Ωlims), β)
            len = b-a
            w = AutoBZCore.workspace_allocate(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
            integrand = OpticalConductivityIntegrand(bz, kalg, w; Σ, β, abstol=abstol/len, reltol)
            solver = IntegralSolver(integrand, AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg; abstol, reltol)

            cnt::Int = 0
            f = BatchFunction() do Ω
                cnt += length(Ω)
                dat = @timed batchsolve(solver, paramzip(; Ω), SMatrix{ndim,ndim,typeof(complex(prec(σatol))),ndim^2}; nthreads=batchthreads)
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(Ω) dat.time
                dat.value
            end

            fn[id] = stats = @timed hchebinterp(f, map(prec, Ωlims)...; atol=σatol, rtol=σrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end

# just do the bz integral
function interpolateconductivityk(; ωlims, Ω, μoffset=zero(Ω),
    io=stdout, verb=true, cachepath=pwd(), nworkers=1, kws...)

    (; t, t′, Δ, ndim, kalg, falg, σatol, σrtol, tolratio, vcomp, bzkind, prec, gauge, coord) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)
    shift!(h, μ)

    id = string((; t, t′, Δ, ndim, η, β, ωlims, Ω, μ, μoffset, kalg, falg, σatol, σrtol, tolratio, vcomp, bzkind, prec, gauge, coord))

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
            integrand = OpticalConductivityIntegrand(bz, kalg, w; Σ, β, Ω=prec(Ω), μ=prec(μoffset), abstol, reltol)

            cnt::Int = 0
            f = BatchFunction() do ω
                cnt += length(ω)
                dat = @timed integrand.(ω, Ref(AutoBZ.CanonicalParameters()))
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
    io=stdout, verb=true, cachepath=pwd(), nworkers=1, kws...)

    (; t, t′, Δ, ndim, kalg, falg, σatol, σrtol, tolratio, vcomp, bzkind, prec, gauge, coord) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)
    shift!(h, μ)

    id = string((; t, t′, Δ, ndim, η, β, μ, Ω, μoffset, kalg, falg, σatol, σrtol, tolratio, vcomp, bzkind, prec, gauge, coord))

    return jldopen(joinpath(cachepath, "cache-conductivity-w.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Interpolating conductivity k integrand to add to cache" id

            hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
            Σ = EtaSelfEnergy(η)
            atol = σatol/det(bz.B)  # corresponding tolerance of k integrand
            abstol = atol/tolratio/nsyms(bz) # tighter tolerance for integration than interpolation
            reltol = σrtol/tolratio
            w = AutoBZCore.workspace_allocate(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
            integrand = OpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), falg, w; Σ, β, Ω=prec(Ω), μ=prec(μoffset), abstol, reltol)

            cnt::Int = 0
            f = BatchFunction() do k
                cnt += length(k)
                dat = @timed integrand.(k, Ref(AutoBZ.CanonicalParameters()))
                verb && @printf io "\t %5i points sampled in %.3e s\n" length(k) dat.time
                dat.value
            end

            fn[id] = stats = @timed hchebinterp(f, map(zero, period(hv)), period(hv); atol, rtol=σrtol)
            verb && @printf io "Done interpolating after %.3e s, %5i sample points\n" stats.time cnt
        end
        return fn[id].value
    end

end
