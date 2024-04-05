function fig3(; colormap=:thermal, lims_y=(0,100), inset_lims_x=nothing, inset_lims_y=nothing, kws...)
    (; scalar_func, scalar_text, series_T, lims_Ω, N_Ω, unit_σ, factor_σ, nsp, ndim, config_vcomp, interp_Ω) = merge(default, NamedTuple(kws))

    series_Ω = range(lims_Ω..., length=N_Ω)
    unit_Ω = unit(eltype(series_Ω))
    fig = Figure(;
        resolution=(3200,2400),
        # figure_padding = 100
        linewidth=6,
        fontsize=100,
    )
    ax = Axis(fig[1,1];
        # title="Fermi liquid optical conductivity",
        xlabel=L"$\Omega$ %$(_unit_Lstr(unit_Ω))",
        xticks=(unique!(Iterators.flatten((getproperty.(config_vcomp, :Ω), lims_Ω)) ./ unit_Ω), [string(isinteger(s) ? Int(s) : s) for s in unique!(Iterators.flatten((getproperty.(config_vcomp, :Ω), lims_Ω)) ./ unit_Ω)]),
        ylabel=L"%$(scalar_text) %$(_unit_Lstr(unit_σ))",
        limits=(lims_Ω ./ unit_Ω, lims_y),
        spinewidth = 4,
        xlabelsize = 100,
        xticksize = 15,
        xtickwidth = 4,
        ylabelsize = 112,
        yticksize = 15,
        ytickwidth = 4,
    )
    min_Ω, i = findmin(series_Ω)
    inset = inset_axis!(fig, ax;
        extent = (0.45, 0.75, 0.65, 0.95),
        xlabel=L"$T$",
        xticks = LogTicks(WilkinsonTicks(3, k_min = 3)),
        xscale=log2,
        xticksize = 15,
        xtickwidth = 4,
        yticks = LogTicks(WilkinsonTicks(3, k_min = 3, k_max=3)),
        ylabel=L"%$(scalar_text)$_{DC}$",
        yscale=log10,
        yticksize = 15,
        ytickwidth = 4,
        spinewidth = 4,
        limits = (inset_lims_x, inset_lims_y),
    )

    Tmax = maximum(series_T)
    Tmin = minimum(series_T)
    unit_T = unit(Tmax)
    colors = cgrad(colormap)
    interp_Ω || error("not implemented")
    T1 = maximum(series_T)
    μ1, = findchempot(; kws..., T=T1)
    _, _, initdiv = conductivity_interp(; kws..., T=T1, μ=μ1) # unroll first calc
    for T in reverse(sort(collect(series_T)))
        μ, = findchempot(; kws..., T)
        data_σ = if interp_Ω
            σ, _, initdiv = conductivity_interp(; kws..., T, μ, initdiv)
            map(Ω -> σ(; Ω), series_Ω)
        else
            error("not implemented")
        end .|> scalar_func .|> σ -> σ*(nsp*factor_σ/(2pi)^ndim/unit_σ) .|> upreferred
        lines!(ax, series_Ω ./ unit_Ω,  data_σ, label=L"%$(ustrip(T))", color=colors[1-log(Tmax/T)/log(Tmax/Tmin)])
        scatter!(inset, T ./ unit_T, data_σ[i], color=colors[1-log(Tmax/T)/log(Tmax/Tmin)], markersize = 40)
    end
    axislegend(ax, ax, L"$T$ (%$(unit_T))";
        framewidth=4,
        width = Relative(0.2),
        patchsize = (100f0, 100f0),
        patchlabelgap = 15,
    )
    lines!(inset, range(1.5*Tmin/unit_T, Tmax/unit_T, length=30), t -> 1e7*t^-2;
        linewidth=8,
        color=:black
    )
    text!(inset, 2^6.3, 2e3, text=L"T⁻²")
    return fig
end
