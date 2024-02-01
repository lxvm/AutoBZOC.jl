function fig3(; colormap=:thermal, lims_y=(0,100), kws...)
    (; scalar_func, scalar_text, series_T, lims_Ω, N_Ω, unit_σ, factor_σ, nsp, ndim, config_vcomp, interp_Ω) = merge(default, NamedTuple(kws))

    series_Ω = range(lims_Ω..., length=N_Ω)
    unit_Ω = unit(eltype(series_Ω))
    fig = Figure(resolution=(800,600))
    ax = Axis(fig[1,1], title="Fermi liquid optical conductivity",
        xlabel=L"$\Omega$ %$(_unit_Lstr(unit_Ω))", xticks=unique!(Iterators.flatten((getproperty.(config_vcomp, :Ω), lims_Ω)) ./ unit_Ω),
        ylabel=L"%$(scalar_text) %$(_unit_Lstr(unit_σ))",
        limits=(lims_Ω ./ unit_Ω, lims_y),
    )
    min_Ω, i = findmin(series_Ω)
    inset = inset_axis!(fig, ax; extent = (0.45, 0.75, 0.65, 0.95), xlabel="T ($(unit(eltype(series_T))))", ylabel="σ(Ω=$(min_Ω)) ($(unit_σ))", xscale=log10, yscale=log10)

    Tmax = maximum(series_T)
    unit_T = unit(Tmax)
    colors = cgrad(colormap)
    for T in reverse(sort(collect(series_T)))
        μ, = findchempot(; kws..., T)
        data_σ = if interp_Ω
            σ, = conductivity_interp(; kws..., T, μ)
            map(Ω -> σ(; Ω), series_Ω)
        else
            error("not implemented")
        end .|> scalar_func .|> σ -> σ*(nsp*factor_σ/(2pi)^ndim/unit_σ) .|> upreferred
        lines!(ax, series_Ω ./ unit_Ω,  data_σ, label="T=$T", color=colors[T/Tmax])
        scatter!(inset, T ./ unit_T, data_σ[i], color=:black)
    end
    axislegend(ax)
    return fig
end
