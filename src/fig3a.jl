# using CairoMakie
using AutoBZ
using LinearAlgebra
using Brillouin: interpolate
using ColorSchemes, Colors
using LaTeXStrings

function fig3a(; alpha=1.0, alpha_ramp = range(0, 1, length=256), ylims=(0, 50),
    cache_file_interp_cond_k="cache-interp-cond-k.jld2", cache_file_interp_cond_ω="cache-interp-cond-ω.jld2",
    cache_file_values_cond_k="cache-values-cond-k.jld2", cache_file_values_cond_ω="cache-values-cond-ω.jld2",
    kws...)
    (; scalar_func, scalar_text, kpath, N_k, N_ω, N_Ω, interp_k, interp_ω, interp_Ω, T, lims_Ω, lims_ω, unit_σ, factor_σ, nsp, ndim, prec, cache_dir, nthreads, auxfun, config_vcomp, atol_σ, rtol_σ, interp_tolratio) = merge(default, NamedTuple(kws))

    fig = Figure(resolution=(800,1000))

    N_vcomp = sum(v -> v.plot_density, config_vcomp)

    β = invtemp(; kws..., T)
    μ, = findchempot(; kws..., T)

    kp = kpath(; kws...)
    kpi = interpolate(kp, N_k)
    # TODO lift the assumption that the k-path is one segment
    kps = KPathSegment(kpi.basis, only(kpi.kpaths), only(kpi.labels), kpi.setting)
    kloc = cumdists(kps.kpath)

    series_Ω = range(prec.(lims_Ω)..., length=N_Ω)
    unit_Ω = unit(eltype(series_Ω))

    ax = Axis(fig[1,1];
        title="Orbital structure of optical conductivity",
        xlabel=L"$\Omega$ %$(_unit_Lstr(unit_Ω))",
        xticks=unique!(Iterators.flatten((getproperty.(config_vcomp, :Ω), lims_Ω)) ./ unit_Ω),
        ylabel=L"%$(scalar_text) %$(_unit_Lstr(unit_σ))",
        limits=(lims_Ω ./ unit_Ω, ylims),
        ygridvisible=false,
    )

    alphabet = 'a':'z'
    Legend(fig[1,2],
        [LineElement(; color) for (; color) in config_vcomp],
        [label for (;label) in config_vcomp],
        string(alphabet[1]),
        width = Relative(1),
    )

    i = 0
    for (; vcomp, label, color, densitycolormap, Ω, plot_trace, plot_density) in config_vcomp
        if plot_trace
            data_σ = if interp_Ω
                σ, = conductivity_interp(; μ, T, kws..., vcomp)
                map(Ω -> σ(; Ω), series_Ω)
            else
                conductivity_batchsolve(; μ, T, kws..., vcomp, series_Ω)[1]
            end .|> scalar_func

            lines!(ax, series_Ω ./ unit_Ω, @show(upreferred.((nsp*factor_σ/(2pi)^ndim/unit_σ) .* data_σ)); color, label)
        end
        if plot_density
            i += 1
            densitycolormap = !isnothing(densitycolormap) ? densitycolormap :
                Colors.alphacolor.(Makie.to_colormap(cgrad([color, color], 100)), alpha_ramp)
            lb_ω, = AutoBZ.fermi_window_limits(Ω, β; rtol=1e-2)
            lims_ω = (lb_ω, -lb_ω)
            series_ω = range(prec.(lims_ω)..., length=N_ω)
            unit_ω = unit(eltype(series_ω))

            ticks_k = (kloc[collect(keys(kps.label))], map(string, values(kps.label)))
            ticks_ω = if iszero(Ω)
                ([-4/β, zero(1/β), 4/β] .* unit(β), [L"-4/\beta", L"0", L"4/\beta"])
            else
                ([-Ω, zero(Ω), Ω] ./ unit_Ω, [L"-\Omega", L"0", L"\Omega"])
            end

            Γ, = transport_solver(; μ, kws..., vcomp)
            σ = FourierIntegrand(Γ.f.w, Γ.f.f, prec(Ω)) do k, Γ, Ω, ω
                scalar_func(fermi_window(β, ω, Ω)*Γ(k, (; ω₁=ω, ω₂=ω+Ω)))
            end

            ax_vcomp = Axis(fig[2i+1,1];
                xlabel=L"Spectral density of %$(scalar_text) at $\Omega$ = %$Ω, T = %$T",
                ylabel=L"$\omega$ %$(_unit_Lstr(unit_ω))",
                xticks=ticks_k,
                yticks=ticks_ω,
                limits = (extrema(kloc), lims_ω ./ unit_ω),
            )

            kpathinterpplot!(ax_vcomp, kps, series_ω, σ; alpha, densitycolormap)
            AutoBZ.shift!(σ.w.series, μ)
            kpathinterpplot!(ax_vcomp, kps, AutoBZ.parentseries(σ.w.series))

            prec_v = x -> map(prec, x)
            data_σ_k = if interp_k
                # problem with nearly-degenerate eigenvectors is that they are noisy
                error("not implemented")
                # TODO break up the kpath into its segments and interpolate each
            else
                σ_k, info_k = conductivity_solver(; kws..., T, μ, vcomp,
                                            bandwidth_bound=Ω, choose_kω_order=(alg...)->true,
                                            lims_Σ=fermi_window_limits(Ω, β))
                hv_k = FourierValue.(prec_v.(kpi), σ_k.f.w.series.(prec_v.(kpi)))
                p_k = merge.(paramproduct(; Ω=prec(Ω), hv_k), Ref(getfield(σ_k.f.f.p, :kwargs)))
                id_k = string((; info_k..., quad_σ_k=nothing, p=hash(p_k)))

                cache_path = joinpath(cache_dir, cache_file_values_cond_k)
                # there is no api to access the inner integral so this is inherently unstable
                # would need a nested integrand type to accomplish that
                cache_batchsolve(σ_k.f.f.f.solver, p_k, cache_path, id_k, nthreads)
            end .|> scalar_func

            ax_vcomp_k = Axis(fig[2i,1];
                ylabel=L"$\int$ %$(scalar_text) $\mathrm{d} \omega$",
                limits=(extrema(kloc), nothing),
                xticks=ticks_k[1],
                yticklabelsvisible=false,
            )
            linkxaxes!(ax_vcomp_k, ax_vcomp)
            hidexdecorations!(ax_vcomp_k, ticks = false, grid = false)

            lines!(ax_vcomp_k, kloc, data_σ_k ./ maximum(data_σ_k); color)


            a, b = AutoBZ.fermi_window_limits(Ω, β)
            atol_σ_ω = atol_σ/(b-a)

            data_σ_ω = if interp_ω
                Γ, info_ω = transport_solver(; kws..., T, μ, vcomp, atol_Γ=atol_σ_ω/interp_tolratio/AutoBZ.fermi_window_maximum(β, Ω), rtol_Γ=rtol_σ/interp_tolratio)
                id_ω = string((; info_ω..., interp_tolratio, lims_ω))

                cache_path = joinpath(cache_dir, cache_file_interp_cond_ω)
                σ_ω = cache_hchebinterp(map(prec, lims_ω)..., atol_σ_ω, rtol_σ, cache_path, id_ω) do ω
                    batchsolve(Γ, paramzip(; ω₁=ω, ω₂=ω .+ prec(Ω)); nthreads) .* fermi_window.(β, ω, prec(Ω))
                end
                map(σ_ω, series_ω)
            else
                Γ, info_ω = transport_solver(; kws..., μ, vcomp, atol_Γ=atol_σ_ω/AutoBZ.fermi_window_maximum(β, Ω), rtol_Γ=rtol_σ)
                cache_path = joinpath(cache_dir, cache_file_values_cond_ω)
                series_ω = range(map(prec, lims_ω)...; length=N_ω)
                p_ω = paramzip(; ω₁=series_ω, ω₂=series_ω .+ prec(Ω))
                id_ω = string((; info_ω..., ω=hash(p_ω)))
                cache_batchsolve(Γ, p_ω, cache_path, id_ω, nthreads) .* fermi_window.(β, series_ω, prec(Ω))
            end .|> scalar_func


            ax_vcomp_ω = Axis(fig[2i+1,2];
                xlabel=L"$\int$ %$(scalar_text) $\mathrm{dk}$",
                xticklabelsvisible=false,
                yticks=ticks_ω[1],
                limits = (nothing, lims_ω ./ unit_ω),
            )
            linkyaxes!(ax_vcomp_ω, ax_vcomp)
            hideydecorations!(ax_vcomp_ω, ticks = false, grid = false)
            lines!(ax_vcomp_ω, data_σ_ω ./ maximum(data_σ_ω), series_ω ./ unit_ω; color)

            Legend(fig[2i,2],
                [LineElement(; color), LineElement(; color=:black)],
                [label, "spectrum"],
                string(alphabet[i+1]),
                width = Relative(1),
                height = Relative(1),
            )

        end
    end

    colsize!(fig.layout, 1, Relative(4/5))
    colsize!(fig.layout, 2, Relative(1/5))

    rowsize!(fig.layout, 1, Relative(1/3))
    i = 0
    for (; plot_density) in config_vcomp
        plot_density || continue
        i += 1
        rowsize!(fig.layout, 2i,    Relative(2/5/N_vcomp*2/3))
        rowsize!(fig.layout, 2i+1,  Relative(3/5/N_vcomp*2/3))
    end

    return fig
end
