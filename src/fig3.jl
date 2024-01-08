function fig3(; colormap=:thermal, lims_y=(0,100), kws...)
    (; scalar_func, scalar_text, series_T, lims_Ω, N_Ω, unit_σ, factor_σ, nsp, ndim, config_vcomp) = merge(default, NamedTuple(kws))

    series_Ω = range(lims_Ω..., length=N_Ω)
    unit_Ω = unit(eltype(series_Ω))
    fig = Figure(resolution=(800,600))
    ax = Axis(fig[1,1], title="Fermi liquid optical conductivity",
        xlabel="Ω ($(unit_Ω))", xticks=unique!(Iterators.flatten((getproperty.(config_vcomp, :Ω), lims_Ω)) ./ unit_Ω),
        ylabel="$(scalar_text) ($(unit_σ))",
        limits=(lims_Ω ./ unit_Ω, lims_y),
    )
    inset = inset_axis!(fig, ax; extent = (0.45, 0.75, 0.65, 0.95), xlabel="T ($(unit(eltype(series_T))))", ylabel="σ(Ω=0) ($(unit_σ))", xscale=log10, yscale=log10)

    Tmax = maximum(series_T)
    unit_T = unit(Tmax)
    colors = cgrad(colormap)
    for T in reverse(sort(collect(series_T)))
        μ, = findchempot(; kws..., T)
        σ, = conductivity_interp(; kws..., T, μ)
        data_σ = map(Ω -> scalar_func(σ(; Ω)), series_Ω)
        lines!(ax, series_Ω ./ unit_Ω,  upreferred.((nsp*factor_σ/(2pi)^ndim/unit_σ) .* data_σ), label="T=$T", color=colors[T/Tmax])
        scatter!(inset, T ./ unit_T, upreferred((nsp*factor_σ/(2pi)^ndim/unit_σ) * scalar_func(σ(; Ω=zero(eltype(series_Ω))))), color=:black)
    end
    axislegend(ax)
    return fig
end
