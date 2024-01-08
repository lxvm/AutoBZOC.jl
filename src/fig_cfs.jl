using LaTeXStrings

function fig_cfs(; colormap=:Spectral, ylims=(0,10), kws...)

    (; series_Δ, lims_Ω, N_Ω, unit_σ, factor_σ, nsp, ndim, interp_Ω, config_vcomp) = merge(default, NamedTuple(kws))

    series_Ω = range(lims_Ω..., length=N_Ω)
    unit_Ω = unit(eltype(series_Ω))
    μ, = findchempot(; kws...)

    fig = Figure(resolution=(1600,600))
    ax1 = Axis(fig[1,1];
        title="Fermi liquid optical conductivity",
        xlabel="Ω ($(unit_Ω))",
        xticks=unique!(Iterators.flatten((getproperty.(config_vcomp, :Ω), lims_Ω)) ./ unit_Ω),
        ylabel=L"$\sigma_{xx}$ (%$(unit_σ))",
        limits=(ustrip.(lims_Ω), ylims),
    )
    ax2 = Axis(fig[1,2];
        title="Fermi liquid optical conductivity",
        xlabel="Ω ($(unit_Ω))",
        xticks=unique!(Iterators.flatten((getproperty.(config_vcomp, :Ω), lims_Ω)) ./ unit_Ω),
        ylabel=L"$\sigma_{yy}$ (%$(unit_σ))",
        limits=(ustrip.(lims_Ω), ylims),
    )

    Δmin = minimum(series_Δ)
    Δmax = maximum(series_Δ)
    cmap = Δ -> Δ >= zero(Δ) ? (1 + Δ/Δmax)/2 : (1 - Δ/Δmin)/2 # keep Δ=0 at center of colorscheme and linear to limits
    colors = cgrad(colormap)

    polx = σ -> real(σ[1,1])
    poly = σ -> real(σ[2,2])
    vcomp = Inter()
    for Δ in sort(collect(series_Δ))[1:2]
        data_σ = if interp_Ω
            σ, = conductivity_interp(; μ, kws..., vcomp, Δ)
            map(Ω -> σ(; Ω), series_Ω)
        else
            conductivity_batchsolve(; μ, T, kws..., vcomp, Δ, series_Ω)[1]
        end
        lines!(ax1, series_Ω ./ unit_Ω, upreferred.((nsp*factor_σ/(2pi)^ndim/unit_σ) .* polx.(data_σ)); label="Δ=$Δ", color=colors[cmap(Δ)])
        lines!(ax2, series_Ω ./ unit_Ω, upreferred.((nsp*factor_σ/(2pi)^ndim/unit_σ) .* poly.(data_σ)); label="Δ=$Δ", color=colors[cmap(Δ)])
    end

    Legend(fig[1,3],
        map(Δ -> LineElement(color=colors[cmap(Δ)]), series_Δ),
        map(Δ -> "Δ=$Δ", series_Δ),
    )
    return fig
end
