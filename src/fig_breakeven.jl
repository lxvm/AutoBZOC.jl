function do_fig_breakeven(bench_func, xlabel, ylabels...; config_quad_breakeven, config_scaling_breakeven, kws...)
    fig = Figure(resolution=(800,1000))
    axs = map(enumerate(ylabels)) do (i, ylabel)
        Axis(fig[i,1];
            xlabel,
            ylabel,
            xscale = log10,
            yscale = log10,
        )
    end

    for (; config_bench, series_T, series_atol_σ, plot_kws) in config_quad_breakeven
        x, data... = bench_func(config_bench, series_T, series_atol_σ, )
        for (ax, dset) in zip(axs, data)
            scatter!(ax, x, dset; plot_kws...)
        end
    end
    for (; x, fun, factors, plot_kws) in config_scaling_breakeven
        for (ax, factor) in zip(axs, factors)
            lines!(ax, x, fun.(x) .* factor ; plot_kws...)
        end
    end
    alphabet = 'a':'z'
    map(((i, ax),) -> Legend(fig[i,1], ax, string(alphabet[i]); tellheight=false, tellwidth=false, halign=:left, valign=:bottom, margin=(10,10,10,10)), enumerate(axs))

    return fig
end

function fig_breakeven(; getpart=getval, kws...)
    (; selfenergy) = merge(default, NamedTuple(kws))
    do_fig_breakeven(L"$\eta$ (eV)", "Wall clock time (s)", "# integrand evaluations"; kws...) do config_bench, series_T, series_atol_σ
        x = Float64[]
        tdat = Float64[]
        ndat = Int[]
        # vdat = prec[] # this is mean to be a measure of the conditioning of
        # the problem, i.e. if there is a scaling between atol and η
        for (T, atol_σ) in zip(series_T, series_atol_σ)
            Σ, = selfenergy(; kws..., T)
            push!(x, AutoBZ.sigma_to_eta(Σ(0.0u"eV"))/u"eV")
            μ, = findchempot(; kws..., T)
            stats, = benchmark_conductivity(; kws..., config_bench..., atol_σ, T, μ)
            @show norm(getval(stats.min.value.sol))
            push!(tdat, stats.min.time)
            push!(ndat, stats.min.value.numevals)
            haskey(stats.min.value, :npt) && @show stats.min.value.npt
            # push!(vdat, norm(getpart(first(stats.samples).value)))
        end
        return x, tdat, ndat#, vdat
    end
end

function fig_breakeven_trgloc(; config_quad_breakeven_trgloc, config_scaling_breakeven_trgloc, kws...)
    (; selfenergy, prec) = merge(default, NamedTuple(kws))
    do_fig_breakeven("η (eV)", "Wall clock time (s)", "# integrand evaluations", "DOS (eV⁻¹)";
        config_quad_breakeven=config_quad_breakeven_trgloc, kws...) do config_bench, series_T
        x = Float64[]
        tdat = Float64[]
        ndat = Int[]
        vdat = prec[] # this is mean to be a measure of the conditioning of
        # the problem, i.e. if there is a scaling between atol and η
        for T in series_T
            Σ, = selfenergy(; kws..., T)
            push!(x, upreferred(AutoBZ.sigma_to_eta(Σ(0.0u"eV"))/u"eV"))
            μ, V, = findchempot(; kws..., T)
            stats, = benchmark_trgloc(; kws..., config_bench..., T, μ)
            push!(tdat, stats.min.time)
            push!(ndat, stats.min.numevals)
            push!(vdat, norm(first(stats.samples).value/V*u"eV"))
        end
        return x, tdat, ndat, vdat
    end
end

function fig_breakeven_log(; getpart=getval, kws...)
    (; selfenergy) = merge(default, NamedTuple(kws))
    do_fig_breakeven(L"$|\log(\eta)|$", "Wall clock time (s)", "# integrand evaluations"; kws...) do config_bench, series_T, series_atol_σ
        x = Float64[]
        tdat = Float64[]
        ndat = Int[]
        # vdat = prec[] # this is mean to be a measure of the conditioning of
        # the problem, i.e. if there is a scaling between atol and η
        for (T, atol_σ) in zip(series_T, series_atol_σ)
            Σ, = selfenergy(; kws..., T)
            push!(x, AutoBZ.sigma_to_eta(Σ(0.0u"eV"))/u"eV")
            μ, = findchempot(; kws..., T)
            stats, = benchmark_conductivity(; kws..., config_bench..., atol_σ, T, μ)
            @show norm(getval(stats.min.value.sol))
            push!(tdat, stats.min.time)
            push!(ndat, stats.min.value.numevals)
            haskey(stats.min.value, :npt) && @show stats.min.value.npt
            # push!(vdat, norm(getpart(first(stats.samples).value)))
        end
        return abs.(log.(x)), tdat, ndat#, vdat
    end
end
