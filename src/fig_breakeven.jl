function do_fig_breakeven(bench_func, xlabel, ylabels...; config_quad_breakeven, config_scaling_breakeven, kws...)
    fig = Figure(;
        resolution=(2400,3000),
        fontsize=80,
        linewidth=8,
    )
    axs = map(enumerate(ylabels)) do (i, (; ylabel, limits))
        Axis(fig[i,1];
            xlabel,
            ylabel,
            xscale = log10,
            yscale = log10,
            spinewidth = 4,
            xlabelsize = 90,
            xticksize = 15,
            xtickwidth = 4,
            ylabelsize = 90,
            yticksize = 15,
            ytickwidth = 4,
            limits
        )
    end

    insets = map(axs) do ax
        #=
        inset_axis!(fig, ax;
            extent = (0.65, 0.95, 0.65, 0.95),
            xlabel=L"$|\log(\eta)|$",
            xticks = LogTicks(WilkinsonTicks(3, k_min = 3)),
            xscale=log10,
            xticksize = 15,
            xtickwidth = 4,
            yticks = LogTicks(WilkinsonTicks(3, k_min = 3, k_max=3)),
            # ylabel=L"%$(scalar_text)$_{DC}$",
            yscale=log10,
            yticksize = 15,
            ytickwidth = 4,
            spinewidth = 4,
        )
        =#
    end

    for (; config_bench, series_T, series_atol_σ, plot_kws) in config_quad_breakeven
        x, data... = bench_func(config_bench, series_T, series_atol_σ, )
        for (ax, inset, dset) in zip(axs, insets, data)
            scatter!(ax, x, dset; markersize = 40, plot_kws...)
            # scatter!(inset, log10.(x), dset; markersize = 40, plot_kws...)
        end
    end
    for (; x, fun, factors, plot_kws) in config_scaling_breakeven
        for (ax, inset, factor) in zip(axs, insets, factors)
            lines!(ax, x, fun.(x) .* factor ; plot_kws...)
            # lines!(inset, log10.(x), fun.(x) .* factor ; plot_kws...)
        end
    end
    alphabet = 'a':'z'
    map(((i, ax),) -> Legend(fig[i,1], ax, string(alphabet[i]); tellheight=false, tellwidth=false, halign=:left, valign=:bottom, margin=(20,20,20,20),
    patchsize = (100f0, 100f0), patchlabelgap = 15,
    framevisible=false,), enumerate(axs))

    return fig
end

function fig_breakeven(; getpart=getval, limits_t=(nothing, nothing), limits_n=(nothing, nothing), kws...)
    (; selfenergy, chempot) = merge(default, NamedTuple(kws))
    do_fig_breakeven(L"$\eta$ (eV)",
        (; ylabel="Wall clock time (s)", limits=limits_t),
        (; ylabel="# integrand evaluations", limits=limits_n)
        ; kws...) do config_bench, series_T, series_atol_σ
        x = Float64[]
        tdat = Float64[]
        ndat = Int[]
        # vdat = prec[] # this is mean to be a measure of the conditioning of
        # the problem, i.e. if there is a scaling between atol and η
        for (T, atol_σ) in zip(series_T, series_atol_σ)
            Σ, = selfenergy(; kws..., T)
            push!(x, AutoBZ.sigma_to_eta(Σ(0.0u"eV"))/u"eV")
            μ, = chempot(; kws..., T)
            stats, = benchmark_conductivity(; kws..., config_bench..., atol_σ, T, μ)
            @show norm(getval(stats.min.value.sol))
            push!(tdat, stats.min.time)
            push!(ndat, stats.min.value.numevals)
            haskey(stats.min.value, :npt) && @show stats.min.value.npt
            # push!(vdat, norm(getpart(first(stats.samples).value)))
        end
        @show x tdat ndat
        return x, tdat, ndat#, vdat
    end
end

function fig_breakeven_trgloc(; config_quad_breakeven_trgloc, config_scaling_breakeven_trgloc, kws...)
    (; selfenergy, prec, chempot) = merge(default, NamedTuple(kws))
    do_fig_breakeven("η (eV)", "Wall clock time (s)", "# integrand evaluations", "DOS (eV⁻¹)";
        config_quad_breakeven=config_quad_breakeven_trgloc,
        config_scaling_breakeven=config_scaling_breakeven_trgloc,
        kws...) do config_bench, series_T, series_atol
        x = Float64[]
        tdat = Float64[]
        ndat = Int[]
        vdat = prec[] # this is mean to be a measure of the conditioning of
        # the problem, i.e. if there is a scaling between atol and η
        for T in series_T
            Σ, = selfenergy(; kws..., T)
            push!(x, upreferred(AutoBZ.sigma_to_eta(Σ(0.0u"eV"))/u"eV"))
            μ, V, = chempot(; kws..., T)
            stats, = benchmark_trgloc(; kws..., config_bench..., T, μ)
            push!(tdat, stats.min.time)
            push!(ndat, stats.min.value.numevals)
            push!(vdat, norm(first(stats.samples).value.sol/V*u"eV"))
        end
        return x, tdat, ndat, vdat
    end
end

function fig_breakeven_log(; getpart=getval, limits_t=(nothing, nothing), limits_n=(nothing, nothing), kws...)
    (; selfenergy, chempot) = merge(default, NamedTuple(kws))
    do_fig_breakeven(L"$|\log(\eta)|$",
    
    (; ylabel="Wall clock time (s)", limits=limits_t),
    (; ylabel="# integrand evaluations", limits=limits_n),
    ; kws...) do config_bench, series_T, series_atol_σ
        x = Float64[]
        tdat = Float64[]
        ndat = Int[]
        # vdat = prec[] # this is mean to be a measure of the conditioning of
        # the problem, i.e. if there is a scaling between atol and η
        for (T, atol_σ) in zip(series_T, series_atol_σ)
            Σ, = selfenergy(; kws..., T)
            push!(x, AutoBZ.sigma_to_eta(Σ(0.0u"eV"))/u"eV")
            μ, = chempot(; kws..., T)

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


function fig_breakeven_trgloc_log(; config_quad_breakeven_trgloc, config_scaling_breakeven_trgloc, kws...)
    (; selfenergy, chempot, prec) = merge(default, NamedTuple(kws))
    do_fig_breakeven("η (eV)", "Wall clock time (s)", "# integrand evaluations"; #, "DOS (eV⁻¹)";
        config_quad_breakeven=config_quad_breakeven_trgloc,
        config_scaling_breakeven=config_scaling_breakeven_trgloc,
        kws...) do config_bench, series_T, series_atol
        x = Float64[]
        tdat = Float64[]
        ndat = Int[]
        vdat = prec[] # this is mean to be a measure of the conditioning of
        # the problem, i.e. if there is a scaling between atol and η
        for T in series_T
            Σ, = selfenergy(; kws..., T)
            push!(x, upreferred(AutoBZ.sigma_to_eta(Σ(0.0u"eV"))/u"eV"))
            μ, V, = chempot(; kws..., T)
            stats, = benchmark_trgloc(; kws..., config_bench..., T, μ)
            push!(tdat, stats.min.time)
            push!(ndat, stats.min.value.numevals)
            push!(vdat, norm(first(stats.samples).value.sol/V*u"eV"))
        end
        return abs.(log10.(x)), tdat, ndat, vdat
    end
end
