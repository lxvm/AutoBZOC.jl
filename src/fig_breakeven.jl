function do_fig_breakeven(bench_func; series_Σ, series_p, kws...)

    (; selfenergy, config_quad_breakeven) = merge(default, NamedTuple(kws))

    algs = config_quad_breakeven
    series_η = map(Σ -> AutoBZ.sigma_to_eta(Σ(0.0u"eV")), series_Σ)
    tdat = Array{Float64,3}(undef, length(series_p), length(series_Σ), length(algs))
    ndat = Array{Int,3}(undef, length(series_p), length(series_Σ), length(algs))
    retcode = Array{Bool,3}(undef, length(series_p), length(series_Σ), length(algs))
    fill!(retcode, false)
    for (k, alg) in enumerate(algs)
        for (j, (η, Σ)) in enumerate(zip(series_η, series_Σ))
            for (i, p) in enumerate(series_p)
                try
                    stats, = bench_func(i, j, alg)
                    tdat[i,j,k] = stats.min.time
                    ndat[i,j,k] = stats.min.numevals
                    retcode[i,j,k] = true
                catch e
                    @info "Benchmark errored" e alg η p
                end
            end
        end
    end
    fig = Figure(resolution=(800,1000))
    ax = Axis(fig[1,1],
        xlabel="η (eV)",
        ylabel="Wall clock time (s)",
        xscale = log10,
        yscale = log10,
        xticks=collect(ustrip.(series_η)),
    )
    # twinax = Axis(fig[1,1], xaxisposition=:top, xlabel="T (K)", xscale = log10, yscale = log10, xticks=ustrip.(collect(series_T)))
    # hidespines!(twinax)
    # hideydecorations!(twinax)
    numevalsax = Axis(fig[2,1],
        xlabel="η (eV)",
        ylabel="# integrand evaluations",
        xscale = log10,
        yscale = log10,
        xticks=collect(ustrip.(series_η)),
    )
    # numevalstwinax = Axis(fig[2,1], xaxisposition=:top, xlabel="T (K)", xscale = log10, yscale = log10, xticks=ustrip.(collect(series_T)))
    # hidespines!(numevalstwinax)
    # hideydecorations!(numevalstwinax)
    for (k, quad_σ_k) in enumerate(algs)
        for (i, p) in enumerate(series_p)
            jmask = retcode[i,:,k]
            any(jmask) || continue
            scatter!(ax, collect(ustrip.(series_η)), tdat[i,jmask,k], label=string(nameof(typeof(quad_σ_k)), "@p=", p))
            # scatter!(twinax, collect(ustrip.(series_T)), tdat[i,jmask,k], label=string(nameof(typeof(quad_σ_k)), "@p=", p))
            scatter!(numevalsax, collect(ustrip.(series_η)), ndat[i,jmask,k], label=string(nameof(typeof(quad_σ_k)), "@p=", p))
            # scatter!(numevalstwinax, collect(ustrip.(series_T)), ndat[i,jmask,k], label=string(nameof(typeof(quad_σ_k)), "@p=", p))
        end
    end
    axislegend(ax)
    axislegend(numevalsax)

    fig
end

function fig_breakeven(; kws...)
    (; selfenergy, series_T, atol_σ, rtol_σ, quadest_σ_k, quadest_σ_ω, config_vcomp) = merge(default, NamedTuple(kws))
    series_Ω = sort!(unique(map(v -> v.Ω, config_vcomp)))
    series_Σ = [selfenergy(; kws..., T)[1] for T in series_T]
    series_μ = [findchempot(; kws..., T)[1] for T in series_T]
    estimates = stack([conductivity_batchsolve(; kws..., T, μ, series_Ω, atol_σ=zero(atol_σ), rtol_σ=one(rtol_σ), quad_σ_k=quadest_σ_k, quad_σ_ω=quadest_σ_ω)[1] for (T, μ) in zip(series_T, series_μ)])
    series_atol_σ = map(norm, estimates)
    do_fig_breakeven(; series_Σ, series_p=series_Ω) do i, j, quad_σ_k
        benchmark_conductivity(; kws..., quad_σ_k, μ=series_μ[j], T=series_T[j], Ω=series_Ω[i], atol_σ=series_atol_σ[i,j]*rtol_σ, rtol_σ=0)
    end
end

function fig_breakeven_trgloc(; kws...)
    (; selfenergy, series_T, atol_g, rtol_g, quad_g_k, config_vcomp) = merge(default, NamedTuple(kws))
    series_ω = sort!(unique(map(v -> v.Ω, config_vcomp)))
    series_μ = [findchempot(; kws..., T)[1] for T in series_T]
    series_Σ = [selfenergy(; kws..., T)[1] for T in series_T]
    # benchmark_trgloc(; kws..., quad_g_k, μ=series_μ[1], T=series_T[1], ω=series_ω[1], atol_g=atol_g, rtol_g=0)
    do_fig_breakeven(; series_Σ, series_p=series_ω) do i, j, quad_g_k
        benchmark_trgloc(; kws..., quad_g_k, μ=series_μ[j], T=series_T[j], ω=series_ω[i], atol_g, rtol_g=0)
    end
end
