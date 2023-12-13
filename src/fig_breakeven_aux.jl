function fig_breakeven_aux(; kws...)

    (; σatol, σrtol, σauxatol, σauxrtol, estkalg, estfalg, Ωseries, Tseries) = merge(default, NamedTuple(kws))

    algs = (IAI(AuxQuadGKJL(order=7)), AutoPTR(nmin=1, nmax=typemax(Int)))
    tdat = Array{Float64,3}(undef, length(Ωseries), length(Tseries), length(algs))
    ndat = Array{Int,3}(undef, length(Ωseries), length(Tseries), length(algs))
    retcode = Array{Bool,3}(undef, length(Ωseries), length(Tseries), length(algs))
    fill!(retcode, false)
    estimates = batchsolveauxconductivity(; kws..., σatol=zero(σatol), σrtol=one(σrtol), σauxatol=zero(σauxatol), σauxrtol=one(σauxrtol), kalg=estkalg, falg=estfalg)
    σatolseries = getproperty.(map(norm, estimates), :val)
    σauxatolseries = getproperty.(map(norm, estimates), :aux)
    for (k, σkalg) in enumerate(algs)
        for (j, T) in enumerate(Tseries)
            for (i, Ω) in enumerate(Ωseries)
                try
                    stats = benchmarkauxconductivity(; kws..., σkalg, T, Ω, σatol=σatolseries[i,j]*σrtol, σrtol=0, σauxrtol=0, σauxatol=σauxatolseries[i,j]*σauxrtol)
                    tdat[i,j,k] = stats.time
                    ndat[i,j,k] = stats.numevals
                    retcode[i,j,k] = true
                catch e
                    @info "Benchmark errored" e σkalg T Ω
                end
            end
        end
    end
    ηseries = map(T -> fermi_liquid_scattering(; kws..., T), Tseries)
    fig = Figure(resolution=(800,1000))
    ax = Axis(fig[1,1],
        xlabel="η (eV)",
        ylabel="Wall clock time (s)",
        xscale = log10,
        yscale = log10,
        xticks=collect(ustrip.(ηseries)),
    )
    twinax = Axis(fig[1,1], xaxisposition=:top, xlabel="T (K)", xscale = log10, yscale = log10, xticks=ustrip.(collect(Tseries)))
    hidespines!(twinax)
    hideydecorations!(twinax)
    numevalsax = Axis(fig[2,1],
        xlabel="η (eV)",
        ylabel="# integrand evaluations",
        xscale = log10,
        yscale = log10,
        xticks=collect(ustrip.(ηseries)),
    )
    numevalstwinax = Axis(fig[2,1], xaxisposition=:top, xlabel="T (K)", xscale = log10, yscale = log10, xticks=ustrip.(collect(Tseries)))
    hidespines!(numevalstwinax)
    hideydecorations!(numevalstwinax)
    for (k, σkalg) in enumerate(algs)
        for (i, Ω) in enumerate(Ωseries)
            jmask = retcode[i,:,k]
            any(jmask) || continue
            scatter!(ax, collect(ustrip.(ηseries)), tdat[i,jmask,k], label=string(nameof(typeof(σkalg)), "@Ω=", Ω))
            scatter!(twinax, collect(ustrip.(Tseries)), tdat[i,jmask,k], label=string(nameof(typeof(σkalg)), "@Ω=", Ω))
            scatter!(numevalsax, collect(ustrip.(ηseries)), ndat[i,jmask,k], label=string(nameof(typeof(σkalg)), "@Ω=", Ω))
            scatter!(numevalstwinax, collect(ustrip.(Tseries)), ndat[i,jmask,k], label=string(nameof(typeof(σkalg)), "@Ω=", Ω))
        end
    end
    axislegend(ax)
    axislegend(numevalsax)

    fig
end

const gcnt_aux = fill(0)

auxiliary_counter_wrapper(auxfun) = (vs, G1, G2) -> begin
    gcnt_aux[] += 1
    return auxfun(vs, G1, G2)
end

function benchmarkauxconductivity(; io=stdout, verb=true, cachepath=pwd(), kws...)

    (; t, t′, Δ, ndim, Ω, σkalg, σfalg, σatol, σrtol, σauxatol, σauxrtol, vcomp, bzkind, prec, gauge, coord, nsample, auxfun) = merge(default, NamedTuple(kws))

    h, bz = t2gmodel(; kws..., gauge=Wannier())
    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    μ = findchempot(; io, verb, cachepath, kws...)
    shift!(h, μ)

    id = string((; t, t′, Δ, ndim, η, β, μ, Ω, σkalg, σfalg, σatol, σrtol, σauxatol, σauxrtol, vcomp, bzkind, prec, gauge, coord, auxfun))

    return jldopen(joinpath(cachepath, "cache-auxconductivity-benchmark.jld2"), "a+") do fn
        if !haskey(fn, id)
            verb && @info "Benchmarking conductivity to add to cache" id

            samples = ntuple(nsample) do n
                gcnt_aux[] = 0
                solver = solverauxconductivity(; μ, bandwidth_bound=Ω, kws..., auxfun=auxiliary_counter_wrapper(auxfun), nworkers=1)
                stats = @timed solver(; Ω=prec(Ω))
                numevals = gcnt_aux[]
                merge(stats, (; numevals))
            end

            fn[id] = samples[findmin(s -> s.time, samples)[2]]

            verb && @printf "Done benchmarking %5i samples with %.3e s average\n" nsample sum(s -> s.time, samples)/nsample
        end
        return fn[id]
    end

end
