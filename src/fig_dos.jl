const ref_dos = Ref{Any}()

function fig_dos(; colormap=:thermal, lims_y=(0,8), kws...)
    (; chempot, series_T, lims_ω, N_ω, nsp, ndim) = merge(default, NamedTuple(kws))

    series_ω = range(lims_ω..., length=N_ω)
    unit_ω = unit(eltype(series_ω))
    fig = Figure(resolution=(800,600))
    ax = Axis(fig[1,1], title="Fermi liquid DOS",
        xlabel="ω ($(unit_ω))",
        ylabel="DOS (eV ^-1)",
        limits=(lims_ω ./ unit_ω, lims_y),
    )

    Tmax = maximum(series_T)
    μ1, = chempot(; kws..., T=Tmax)
    _, _, initdiv = trgloc_interp(; kws..., T=Tmax, μ=μ1) # unroll first calc
    colors = cgrad(colormap)
    for T in reverse(sort(collect(series_T)))
        μ, V, = chempot(; kws..., T)
        g, _, initdiv = trgloc_interp(; kws..., T, μ, initdiv)
        data_ρ = map(ω -> upreferred(-imag(g(; ω))/pi/V * nsp * u"eV"), series_ω)
        lines!(ax, series_ω ./ unit_ω, data_ρ, label="T=$T", color=colors[T/Tmax])
        ref_dos[] = (V, initdiv)
    end
    axislegend(ax)
    return fig
end
