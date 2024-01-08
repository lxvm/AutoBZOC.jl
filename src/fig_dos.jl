function fig_dos(; colormap=:thermal, lims_y=(0,8), kws...)
    (; series_T, lims_ω, N_ω, nsp, ndim) = merge(default, NamedTuple(kws))

    series_ω = range(lims_ω..., length=N_ω)
    unit_ω = unit(eltype(series_ω))
    fig = Figure(resolution=(800,600))
    ax = Axis(fig[1,1], title="Fermi liquid DOS",
        xlabel="ω ($(unit_ω))",
        ylabel="DOS (eV ^-1)",
        limits=(lims_ω ./ unit_ω, lims_y),
    )

    Tmax = maximum(series_T)
    colors = cgrad(colormap)
    for T in reverse(sort(collect(series_T)))
        μ, V, = findchempot(; kws..., T)
        g, = trgloc_interp(; kws..., T, μ)
        data_ρ = map(ω -> upreferred(-imag(g(; ω))/pi/V * nsp/(2pi)^ndim * u"eV"), series_ω)
        lines!(ax, series_ω ./ unit_ω, data_ρ, label="T=$T", color=colors[T/Tmax])
    end
    axislegend(ax)
    return fig
end
