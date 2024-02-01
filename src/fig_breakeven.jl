function do_fig_breakeven(bench_func; series_Σ, series_p, label_p, kws...)
    (; config_quad_breakeven) = merge(default, NamedTuple(kws))

    (; algs, series) = config_quad_breakeven
    series_η = map(Σ -> AutoBZ.sigma_to_eta(Σ(0.0u"eV")), series_Σ)
    dims = map(length, (series_p, series_η, algs))
    tdat = zeros(Float64, dims)
    vdat = zeros(Float64, dims)
    ndat = zeros(Int, dims)
    retcode = zeros(Bool, dims)
    fill!(retcode, false)
    for (k, alg) in enumerate(algs)
        for (j, η) in enumerate(series_η)
            for (i, p) in enumerate(series_p)
                try
                    tdat[i,j,k], ndat[i,j,k], vdat[i,j,k] = bench_func(i, j, alg)
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
        # xticks=collect(ustrip.(series_η)),
        limits=(nothing, extrema(filter(>(0), tdat))),
    )
    # twinax = Axis(fig[1,1], xaxisposition=:top, xlabel="T (K)", xscale = log10, yscale = log10, xticks=ustrip.(collect(series_T)))
    # hidespines!(twinax)
    # hideydecorations!(twinax)
    numevalsax = Axis(fig[2,1],
        xlabel="η (eV)",
        ylabel="# integrand evaluations",
        xscale = log10,
        yscale = log10,
        # xticks=collect(ustrip.(series_η)),
        limits=(nothing, extrema(filter(>(0), ndat))),
    )
    # numevalstwinax = Axis(fig[2,1], xaxisposition=:top, xlabel="T (K)", xscale = log10, yscale = log10, xticks=ustrip.(collect(series_T)))
    # hidespines!(numevalstwinax)
    # hideydecorations!(numevalstwinax)
    tolax = Axis(fig[3,1],
        xlabel="η (eV)",
        ylabel="tolerance/result",
        xscale = log10,
        yscale = log10,
        # xticks=collect(ustrip.(series_η)),
        limits=(nothing, extrema(filter(>(0), vdat))),
    )
    for (k, quad_σ_k) in enumerate(algs)
        for (i, p) in enumerate(series_p)
            jmask = retcode[i,:,k]
            any(jmask) || continue
            x = ustrip.(series_η[jmask])
            scatter!(ax, x, tdat[i,jmask,k], label=string(nameof(typeof(quad_σ_k)), "@$label_p=", p))
            scatter!(numevalsax, x, ndat[i,jmask,k], label=string(nameof(typeof(quad_σ_k)), "@$label_p=", p))
            scatter!(tolax, x, vdat[i,jmask,k], label=string(nameof(typeof(quad_σ_k)), "@$label_p=", p))
        end
    end
    for (; fun, label, factor_t, factor_numevals) in series
        x = ustrip.(series_η)
        lines!(ax, x, fun.(x) .* factor_t; label)
        lines!(numevalsax, x, fun.(x) .* factor_numevals; label)
    end
    axislegend(ax)
    axislegend(numevalsax)
    axislegend(tolax)

    fig
end

function fig_breakeven(; series_Ω, use_estimate=true, getpart=getval, kws...)
    (; selfenergy, series_T, atol_σ, rtol_σ, quadest_σ_k, quadest_σ_ω) = merge(default, NamedTuple(kws))
    series_Σ = [selfenergy(; kws..., T)[1] for T in series_T]
    series_μ = [findchempot(; kws..., T)[1] for T in series_T]
    series_atol_σ = if use_estimate
        estimates = stack([conductivity_batchsolve(; kws..., T, μ, series_Ω, atol_σ=zero(atol_σ), rtol_σ=one(rtol_σ), quad_σ_k=quadest_σ_k, quad_σ_ω=quadest_σ_ω)[1] for (T, μ) in zip(series_T, series_μ)])
        map(norm, estimates)
    else
        fill(atol_σ, length(series_T), length(series_μ))
    end
    do_fig_breakeven(; series_Σ, series_p=series_Ω, label_p="Ω", kws...) do i, j, quad_σ_k
        atol_σ=series_atol_σ[i,j]*rtol_σ
        stats, = benchmark_conductivity(; kws..., quad_σ_k, μ=series_μ[j], T=series_T[j], Ω=series_Ω[i], atol_σ, rtol_σ=0)
        stats.min.time, stats.min.numevals, getpart(atol_σ/norm(first(stats.samples).value))
    end
end

function fig_breakeven_trgloc(; series_ω, kws...)
    (; selfenergy, series_T, atol_g, rtol_g, quad_g_k) = merge(default, NamedTuple(kws))
    series_μ = [findchempot(; kws..., T)[1] for T in series_T]
    series_Σ = [selfenergy(; kws..., T)[1] for T in series_T]
    # benchmark_trgloc(; kws..., quad_g_k, μ=series_μ[1], T=series_T[1], ω=series_ω[1], atol_g=atol_g, rtol_g=0)
    do_fig_breakeven(; series_Σ, series_p=series_ω, label_p="ω", kws...) do i, j, quad_g_k
        stats, = benchmark_trgloc(; kws..., quad_g_k, μ=series_μ[j], T=series_T[j], ω=series_ω[i], atol_g, rtol_g=0)
        stats.min.time, stats.min.numevals, atol_g/norm(first(stats.samples).value)
    end
end
