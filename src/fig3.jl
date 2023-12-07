function fig3(; colormap=:thermal, ylims=(0,100), kws...)

    (; scalarize, scalarize_text, Tseries, Ωlims, Ωintra, Ωinter, NΩ, σudisplay, σufactor, nsp, ndim) = merge(default, NamedTuple(kws))

    Ωs = range(Ωlims..., length=NΩ)
    fig = Figure(resolution=(800,600))
    ax = Axis(fig[1,1], title="Fermi liquid optical conductivity",
        xlabel="Ω ($(unit(eltype(Ωs))))", xticks=ustrip.(unique(append!(collect(Ωlims), [Ωinter, Ωintra]))),
        ylabel="$(scalarize_text) ($(σudisplay))",
        limits=(ustrip.(Ωlims), ylims),
    )

    inset = inset_axis!(fig, ax; extent = (0.45, 0.75, 0.65, 0.95), xlabel="T ($(unit(eltype(Tseries))))", ylabel="σ(Ω=0) ($(σudisplay))", xscale=log10, yscale=log10)

    Tmax = maximum(Tseries)
    colors = cgrad(colormap)
    for T in reverse(sort(collect(Tseries)))
        cond = interpolateconductivity(; kws..., T)
        lines!(ax, ustrip.(Ωs), ustrip.(uconvert.(σudisplay, scalarize.((nsp*σufactor/(2pi)^ndim) .* cond.(Ωs)))), label="T=$T", color=colors[T/Tmax])
        scatter!(inset, ustrip(T), ustrip(uconvert(σudisplay, scalarize((nsp*σufactor/(2pi)^ndim) * cond(zero(Ωintra))))), color=:black)
    end
    axislegend(ax)
    return fig
end
