function fig3_aux(; colormap=:thermal, auxfield=:aux, kws...)

    (; scalarize, scalarize_text, Tseries, Ωlims, Ωintra, Ωinter, NΩ, σauxatol) = merge(default, NamedTuple(kws))

    Ωs = range(Ωlims..., length=NΩ)
    fig = Figure(resolution=(800,600))
    ax = Axis(fig[1,1], title="Fermi liquid auxiliary conductivity",
        xlabel="Ω ($(unit(eltype(Ωs))))", xticks=ustrip.(unique(append!(collect(Ωlims), [Ωinter, Ωintra]))),
        ylabel="$(scalarize_text) ($(unit(σauxatol)))",
        limits=(ustrip.(Ωlims), nothing),
    )
    Tmax = maximum(Tseries)
    colors = cgrad(colormap)
    for T in reverse(sort(collect(Tseries)))
        cond = interpolateauxconductivity(; kws..., T)
        lines!(ax, ustrip.(Ωs), ustrip.(scalarize.(getproperty.(cond.(Ωs), Ref(auxfield)))), label="T=$T", color=colors[T/Tmax])
    end
    axislegend(ax)
    return fig
end
