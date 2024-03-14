function fig_err(; Ω, kws...)
    (; T, config_auxquad, config_reference, series_rtol_σ) = merge(default, NamedTuple(kws))
    dims = (length(series_rtol_σ), length(config_auxquad))
    tdat = zeros(Float64, dims)
    ndat = zeros(Int64, dims)
    edat = zeros(Float64, dims)
    μ, _, info = findchempot(; kws..., T)
    stats_ref, = benchmark_conductivity(; kws..., config_reference..., μ, T, Ω)
    val = first(stats_ref.samples).value
    @info "reference conductivity value" val=val.val

    series_atol_σ = series_rtol_σ .* norm(getval(val))
    for (j, conf) in enumerate(config_auxquad)
        for (i, atol_σ) in enumerate(series_atol_σ)
            stats, = benchmark_conductivity(; kws..., conf..., μ, T, Ω, rtol_σ=0.0, atol_σ=conf.auxfun === nothing ? atol_σ : AuxValue(atol_σ, conf.atol_σ_aux))
            v = first(stats.samples).value
            # @show norm(getaux(v)) norm(getval(v))
            tdat[i,j] = stats.min.time
            ndat[i,j] = stats.min.numevals
            edat[i,j] = norm(getval(v)-getval(val))/norm(getval(val))
        end
    end
    fig = Figure(resolution=(800,1000))
    eax = Axis(fig[1,1];
        title = "norm(σ(Ω=$Ω; T=$T))=$(norm(getval(val))), $(info.model.name), $(info.model.ndim)-dim",
        xlabel="requested error (relative to true value)",
        ylabel="relative error",
        xscale = log10,
        yscale = log10,
        # xticks=collect(ustrip.(series_η)),
        limits=(nothing, (1e-16, 1e0)),
    )
    tax = Axis(fig[2,1],
        xlabel="requested error (relative to true value)",
        ylabel="Wall clock time (s)",
        xscale = log10,
        yscale = log10,
        # xticks=collect(ustrip.(series_η)),
        limits=(nothing, extrema(filter(>(0), tdat))),
    )
    nax = Axis(fig[3,1],
        xlabel="requested error (relative to true value)",
        ylabel="Integrand evaluations",
        xscale = log10,
        yscale = log10,
        # xticks=collect(ustrip.(series_η)),
        limits=(nothing, extrema(filter(>(0), ndat))),
    )
    for (i, (; label, marker)) in enumerate(config_auxquad)
        scatter!(eax, series_rtol_σ, edat[:,i]; label, marker)
        scatter!(tax, series_rtol_σ, tdat[:,i]; label, marker)
        scatter!(nax, series_rtol_σ, ndat[:,i]; label, marker)
    end
    lines!(eax,  series_rtol_σ, series_rtol_σ; label="tolerance", color=:red, linestyle=:dash)
    axislegend(eax; position=:rb)
    axislegend(tax; position=:lb)
    axislegend(nax; position=:lb)
    return fig
end

function fig_err_dos(; ω, kws...)
    (; T, config_reference_dos, config_dosquad, series_rtol_g) = merge(default, NamedTuple(kws))
    dims = (length(series_rtol_g), length(config_dosquad))
    tdat = zeros(Float64, dims)
    ndat = zeros(Int64, dims)
    edat = zeros(Float64, dims)
    μ, V, info = findchempot(; kws..., T)
    stats_ref, = benchmark_trgloc(; kws..., config_reference_dos..., μ, T, ω)
    val = first(stats_ref.samples).value

    series_atol_g = series_rtol_g .* norm(val)
    for (j, conf) in enumerate(config_dosquad)
        for (i, atol_g) in enumerate(series_atol_g)
            stats, = benchmark_trgloc(; kws..., conf..., μ, T, ω, rtol_g=0.0, atol_g)
            v = first(stats.samples).value
            tdat[i,j] = stats.min.time
            ndat[i,j] = stats.min.numevals
            edat[i,j] = norm(getval(v)-getval(val))/norm(getval(val))
        end
    end
    fig = Figure(resolution=(800,1000))
    eax = Axis(fig[1,1];
        title = "norm(DOS(ω=$ω; T=$T))=$(-imag(val)/pi/V), $(info.model.name), $(info.model.ndim)-dim",
        xlabel="requested error (relative to true value)",
        ylabel="relative error",
        xscale = log10,
        yscale = log10,
        yticks=([1e-16, 1e-12, 1e-8, 1e-4, 1e-0], ["1e-16", "1e-12", "1e-8", "1e-4", "1e0"]),
        limits=(nothing, (1e-16, 1e0)),
    )
    tax = Axis(fig[2,1],
        xlabel="requested error (relative to true value)",
        ylabel="Wall clock time (s)",
        xscale = log10,
        yscale = log10,
        limits=(nothing, extrema(filter(>(0), tdat))),
    )
    nax = Axis(fig[3,1],
        xlabel="requested error (relative to true value)",
        ylabel="Integrand evaluations",
        xscale = log10,
        yscale = log10,
        limits=(nothing, extrema(filter(>(0), ndat))),
    )
    for (i, (; label)) in enumerate(config_dosquad)
        scatter!(eax, series_rtol_g, edat[:,i]; label)
        scatter!(tax, series_rtol_g, tdat[:,i]; label)
        scatter!(nax, series_rtol_g, ndat[:,i]; label)
    end
    lines!(eax,  series_rtol_g, series_rtol_g; label="tolerance", color=:red, linestyle=:dash)
    axislegend(eax; position=:rb)
    axislegend(tax; position=:lb)
    axislegend(nax; position=:lb)
    @show series_atol_g
    return fig
end

function fig_err_test(; Ω, kws...)
    (; T, config_testquad, config_reference, series_rtol_σ) = merge(default, NamedTuple(kws))
    dims = (length(series_rtol_σ), length(config_testquad))
    tdat = zeros(Float64, dims)
    ndat = zeros(Int64, dims)
    edat = zeros(Float64, dims)
    μ, _, info = findchempot(; kws..., T)
    stats_ref, = benchmark_conductivity_test(; kws..., config_reference..., μ, T, Ω)
    val = first(stats_ref.samples).value
    @info "reference conductivity value" val=val.v
    for (j, conf) in enumerate(config_testquad)
        for (i, rtol_σ) in enumerate(series_rtol_σ)
            atol_σ = rtol_σ * conf.series_rtol_σ_aux * norm(val.v[end])
            stats, = benchmark_conductivity_test(; kws..., conf..., μ, T, Ω, rtol_σ=0.0, atol_σ)
            v = first(stats.samples).value
            @show v
            # @show norm(getaux(v)) norm(getval(v))
            tdat[i,j] = stats.min.time
            ndat[i,j] = stats.min.numevals
            edat[i,j] = norm(v.v[end]-val.v[end])/norm(val.v[end])
        end
    end
    fig = Figure(resolution=(800,1000))
    eax = Axis(fig[1,1];
        title = "norm(σ(Ω=$Ω; T=$T))=$(norm(val.v[end])), $(info.model.name), $(info.model.ndim)-dim",
        xlabel="requested error (relative to true value)",
        ylabel="relative error",
        xscale = log10,
        yscale = log10,
        # xticks=collect(ustrip.(series_η)),
        limits=(nothing, (1e-16, 1e0)),
    )
    tax = Axis(fig[2,1],
        xlabel="requested error (relative to true value)",
        ylabel="Wall clock time (s)",
        xscale = log10,
        yscale = log10,
        # xticks=collect(ustrip.(series_η)),
        limits=(nothing, extrema(filter(>(0), tdat))),
    )
    nax = Axis(fig[3,1],
        xlabel="requested error (relative to true value)",
        ylabel="Integrand evaluations",
        xscale = log10,
        yscale = log10,
        # xticks=collect(ustrip.(series_η)),
        limits=(nothing, extrema(filter(>(0), ndat)) .* (0.9, 1.1)),
    )
    for (i, (; label, marker)) in enumerate(config_testquad)
        scatter!(eax, series_rtol_σ, edat[:,i]; label, marker)
        scatter!(tax, series_rtol_σ, tdat[:,i]; label, marker)
        scatter!(nax, series_rtol_σ, ndat[:,i]; label, marker)
    end
    lines!(eax,  series_rtol_σ, series_rtol_σ; label="tolerance", color=:red, linestyle=:dash)
    axislegend(eax; position=:rb)
    axislegend(tax; position=:lb)
    axislegend(nax; position=:lb)
    @show val
    return fig
end


function fig_err_only(; Ω, kws...)
    (; T, config_onlyquad, config_reference_only, series_rtol_σ) = merge(default, NamedTuple(kws))
    dims = (length(series_rtol_σ), length(config_onlyquad))
    tdat = zeros(Float64, dims)
    ndat = zeros(Int64, dims)
    edat = zeros(Float64, dims)
    μ, _, info = findchempot(; kws..., T)
    stats_ref, = benchmark_conductivity_only(; kws..., config_reference_only..., μ, T, Ω)
    val = first(stats_ref.samples).value
    @info "reference conductivity value" val
    for (j, conf) in enumerate(config_onlyquad)
        for (i, rtol_σ) in enumerate(series_rtol_σ)
            atol_σ = rtol_σ * norm(val)
            @show rtol_σ
            stats, = benchmark_conductivity_only(; kws..., rtol_σ, μ, T, Ω, atol_σ=conf.auxfun === nothing ? atol_σ : AuxValue(atol_σ, conf.atol_σ_aux), conf...)
            v = first(stats.samples).value
            @show v
            # @show norm(getaux(v)) norm(getval(v))
            tdat[i,j] = stats.min.time
            ndat[i,j] = stats.min.numevals
            edat[i,j] = norm(v-val)/norm(val)
        end
    end
    fig = Figure(resolution=(800,1000))
    eax = Axis(fig[1,1];
        title = "norm(σ(Ω=$Ω; T=$T))=$(norm(val)), $(info.model.name), $(info.model.ndim)-dim",
        xlabel="requested error (relative to true value)",
        ylabel="relative error",
        xscale = log10,
        yscale = log10,
        # xticks=collect(ustrip.(series_η)),
        limits=(nothing, (1e-16, 1e0)),
    )
    tax = Axis(fig[2,1],
        xlabel="requested error (relative to true value)",
        ylabel="Wall clock time (s)",
        xscale = log10,
        yscale = log10,
        # xticks=collect(ustrip.(series_η)),
        limits=(nothing, extrema(filter(>(0), tdat))),
    )
    nax = Axis(fig[3,1],
        xlabel="requested error (relative to true value)",
        ylabel="Integrand evaluations",
        xscale = log10,
        yscale = log10,
        # xticks=collect(ustrip.(series_η)),
        limits=(nothing, extrema(filter(>(0), ndat)) .* (0.9, 1.1)),
    )
    for (i, (; label, marker)) in enumerate(config_onlyquad)
        scatter!(eax, series_rtol_σ, edat[:,i]; label, marker)
        scatter!(tax, series_rtol_σ, tdat[:,i]; label, marker)
        scatter!(nax, series_rtol_σ, ndat[:,i]; label, marker)
    end
    lines!(eax,  series_rtol_σ, series_rtol_σ; label="tolerance", color=:red, linestyle=:dash)
    axislegend(eax; position=:rb)
    axislegend(tax; position=:lb)
    axislegend(nax; position=:lb)
    @show val
    return fig
end
