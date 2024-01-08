using CairoMakie
using AutoBZ
using LinearAlgebra
using Brillouin: interpolate
using ColorSchemes, Colors


function fig3a(; alpha=1.0, alpha_ramp = range(0, 1, length=256), ylims=(0, 50),
    cache_file_interp_cond_k="cache-interp-cond-k.jld2", cache_file_interp_cond_ω="cache-interp-cond-ω.jld2",
    cache_file_values_cond_k="cache-values-cond-k.jld2", cache_file_values_cond_ω="cache-values-cond-ω.jld2",
    kws...)
    (; scalar_func, scalar_text, kpath, N_k, N_ω, N_Ω, interp_k, interp_ω, interp_Ω, T, lims_Ω, lims_ω, unit_σ, factor_σ, nsp, ndim, prec, cache_dir, nthreads, auxfun, config_vcomp, atol_σ, rtol_σ, interp_tolratio) = merge(default, NamedTuple(kws))

    fig = Figure(resolution=(800,1600))

    N_vcomp = sum(v -> v.plot_density, config_vcomp)
    layout_cond = GridLayout(; parent=fig)
    layout_structure = GridLayout(;
        nrow=2*N_vcomp,
        ncol=2,
        parent=fig,
    )

    β = invtemp(; kws..., T)
    μ, = findchempot(; kws..., T)

    series_ω = range(prec.(lims_ω)..., length=N_ω)
    unit_ω = unit(eltype(series_ω))

    kp = kpath(; kws...)
    kpi = interpolate(kp, N_k)
    # TODO lift the assumption that the k-path is one segment
    kps = KPathSegment(kpi.basis, only(kpi.kpaths), only(kpi.labels), kpi.setting)
    kloc = cumdists(kps.kpath)

    series_Ω = range(prec.(lims_Ω)..., length=N_Ω)
    unit_Ω = unit(eltype(series_Ω))

    layout_cond[1,1] = ax = Axis(fig;
        title="Orbital structure of optical conductivity",
        xlabel="Ω ($(unit_Ω))",
        xticks=unique!(Iterators.flatten((getproperty.(config_vcomp, :Ω), lims_Ω)) ./ unit_Ω),
        ylabel="$(scalar_text) ($(unit_σ))",
        limits=(lims_Ω ./ unit_Ω, ylims),
    )

    i = 0
    for (; vcomp, label, color, Ω, plot_trace, plot_density) in config_vcomp
        if plot_trace
            data_σ = if interp_Ω
                σ, = conductivity_interp(; μ, T, kws..., vcomp)
                map(Ω -> σ(; Ω), series_Ω)
            else
                conductivity_batchsolve(; μ, T, kws..., vcomp, series_Ω)[1]
            end .|> scalar_func
            lines!(ax, series_Ω ./ unit_Ω, upreferred.((nsp*factor_σ/(2pi)^ndim/unit_σ) .* data_σ); color, label)
        end
        if plot_density
            i += 1
            densitycolormap = Colors.alphacolor.(Makie.to_colormap(cgrad([color, color], 100)), alpha_ramp)

            layout_structure[2i,1] = ax_vcomp = Axis(fig,
                xlabel="Spectral density of $(scalar_text)", ylabel="ω ($unit_ω)",
                xticks=(kloc[collect(keys(kps.label))], map(string, values(kps.label))),
                limits = (extrema(kloc), (1,N_ω))
            )

            Γ, = transport_solver(; μ, kws..., vcomp)
            σ = FourierIntegrand(Γ.f.w, Γ.f.f, prec(Ω)) do k, Γ, Ω, ω
                scalar_func(fermi_window(β, ω, Ω)*Γ(k, (; ω₁=ω, ω₂=ω+Ω)))
            end

            kpathinterpplot!(ax_vcomp, kps, series_ω, σ; alpha, densitycolormap)
            kpathinterpplot!(ax_vcomp, kps, AutoBZ.parentseries(σ.w.series))

            inset = inset_axis!(fig, ax_vcomp;
                extent = (0.15, 0.45, 0.45, 0.75),
                limits = ((1.0,2.5), (-0.05, 0.05)),
                xticklabelsvisible=false,
                xticksvisible=false,
                xgridvisible=false,
                ygridvisible=false,
            )

            kpathinterpplot!(inset, kps, series_ω, σ; alpha, densitycolormap)
            kpathinterpplot!(inset, kps, AutoBZ.parentseries(σ.w.series))

            layout_structure[2i-1,1] = ax_vcomp_k = Axis(fig;
                ylabel="∫ $(scalar_text)(k,ω) dω",
                limits=(extrema(kloc), nothing),
                ygridvisible=false,
                yticks=[0.0],
            )
            linkxaxes!(ax_vcomp_k, ax_vcomp)
            hidexdecorations!(ax_vcomp_k, ticks = false, grid = false)

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
            lines!(ax_vcomp_k, kloc, data_σ_k ./ maximum(data_σ_k); color)


            layout_structure[2i,2] = ax_vcomp_ω = Axis(fig;
                xlabel="∫ $(scalar_text)(k,ω) dk",
                xgridvisible=false,
                xticks=[0.0],
            )
            linkyaxes!(ax_vcomp_ω, ax_vcomp)
            hideydecorations!(ax_vcomp_ω, ticks = false, grid = false)

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

            lines!(ax_vcomp_ω, data_σ_ω ./ maximum(data_σ_ω), series_ω ./ unit_ω; color)

            Legend(layout_structure[2i-1,2],
                [LineElement(; color=:black), LineElement(; color)],
                ["spectrum", label],
            )

        end
    end

    axislegend(ax)

    colsize!(layout_structure, 1, Relative(4/5))
    colsize!(layout_structure, 2, Relative(1/5))
    i = 0
    for (; plot_density) in config_vcomp
        plot_density || continue
        i += 1
        rowsize!(layout_structure, 2i-1, Relative(1/5/N_vcomp))
        rowsize!(layout_structure, 2i,   Relative(4/5/N_vcomp))
    end

    fig.layout[1,1] = layout_cond
    fig.layout[2,1] = layout_structure
    rowsize!(fig.layout, 1, Relative(2/9))
    rowsize!(fig.layout, 2, Relative(7/9))

    return fig
end
