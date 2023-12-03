function fig3(;
    io = stdout,
    verb = true,
    colormap = :thermal,
    scalarize = default.scalarize,
    scalarize_text = default.scalarize_text,
    t = default.t,
    t′ = default.t′,
    Δ = default.Δ,
    nalg = default.nalg,
    natol = default.natol,
    nrtol = default.nrtol,
    μlims = default.μlims,
    Tseries = default.Tseries,
    T₀ = default.T₀,
    Z = default.Z,
    Ωlims = default.Ωlims,
    atol = default.σatol,
    rtol = default.σrtol,
    ν = default.ν,
    nsp = default.nsp,
    falg = default.falg,
    kalg = default.kalg,
    bzkind = default.bzkind,
    tolratio = default.tolratio,
    prec = default.prec,
    NΩ = default.NΩ,
)
    Ωs = range(Ωlims..., length=NΩ)
    fig = Figure(resolution=(800,600))
    ax = Axis(fig[1,1], title="Fermi liquid optical conductivity",
        xlabel="Ω ($(unit(eltype(Ωs))))",
        ylabel="$(scalarize_text) ($(unit(atol)))", yscale=log10,
        limits=(ustrip.(Ωlims), nothing),
    )
    Tmax = maximum(Tseries)
    colors = cgrad(colormap)
    for T in Tseries
        cond = interpolateconductivitykw(vcomp=Whole(), t=t, t′=t′, Δ=Δ, T=T, T₀=T₀, Z=Z, kalg=kalg, falg=falg, natol=natol, nrtol=nrtol, nalg=nalg, bzkind=bzkind,
                μlims=μlims, ν=ν, nsp=nsp, Ωlims=Ωlims, atol=atol, rtol=rtol, io=io, verb=verb, tolratio=tolratio, prec=prec)
        lines!(ax, ustrip.(Ωs), ustrip.(scalarize.(cond.(Ωs))), label="T=$T", color=colors[T/Tmax])
    end
    axislegend(ax)
    return fig
end
