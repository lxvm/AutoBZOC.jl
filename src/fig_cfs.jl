using LaTeXStrings

function fig_cfs(; colormap=:Spectral, ylims=(0,10), kws...)

    (; Δseries, Ωlims, Ωintra, Ωinter, NΩ, σudisplay, σufactor, nsp, ndim) = merge(default, NamedTuple(kws))

    Ωs = range(Ωlims..., length=NΩ)
    fig = Figure(resolution=(1600,600))
    ax1 = Axis(fig[1,1], title="Fermi liquid optical conductivity",
        xlabel="Ω ($(unit(eltype(Ωs))))", xticks=ustrip.(unique(append!(collect(Ωlims), [Ωinter, Ωintra]))),
        ylabel=L"$\sigma_{xx}$ (%$(σudisplay))",
        limits=(ustrip.(Ωlims), ylims),
    )
    ax2 = Axis(fig[1,2], title="Fermi liquid optical conductivity",
        xlabel="Ω ($(unit(eltype(Ωs))))", xticks=ustrip.(unique(append!(collect(Ωlims), [Ωinter, Ωintra]))),
        ylabel=L"$\sigma_{yy}$ (%$(σudisplay))",
        limits=(ustrip.(Ωlims), ylims),
    )

    Δmin = minimum(Δseries)
    Δmax = maximum(Δseries)
    cmap = Δ -> Δ >= zero(Δ) ? (1 + Δ/Δmax)/2 : (1 - Δ/Δmin)/2 # keep Δ=0 at center of colorscheme and linear to limits
    colors = cgrad(colormap)

    polx = σ -> real(σ[1,1])
    poly = σ -> real(σ[2,2])
    for Δ in sort(collect(Δseries))
        cond = interpolateconductivity(; kws..., vcomp=Inter(), Δ)
        lines!(ax1, ustrip.(Ωs), ustrip.(uconvert.(σudisplay, polx.((nsp*σufactor/(2pi)^ndim) .* cond.(Ωs)))), label="Δ=$Δ", color=colors[cmap(Δ)])
        lines!(ax2, ustrip.(Ωs), ustrip.(uconvert.(σudisplay, poly.((nsp*σufactor/(2pi)^ndim) .* cond.(Ωs)))), label="Δ=$Δ", color=colors[cmap(Δ)])
    end

    Legend(fig[1,3],
    map(Δ -> LineElement(color=colors[cmap(Δ)]), Δseries),
    map(Δ -> "Δ=$Δ", Δseries),
    )
    return fig
end
