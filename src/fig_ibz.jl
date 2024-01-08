using GLMakie
using Brillouin
using SymmetryReduceBZ
using Polyhedra: Polyhedron, Mesh, points, polyhedron, vrep
using FourierSeriesEvaluators: workspace_evaluate!, workspace_contract!

function fig_ibz(; kpath=sgnum_path, colormap=:Spectral, N_k=30, sgnum=76, whichsym=13,
    alpharamp = range(0,1,length=256) .^ 4, cache_file_values_cond_k_cfs="cache-values-cond-k-cfs.jld2", kws...)
    (; T, series_Δ, vcomp, ndim, prec, config_vcomp, cache_dir, nthreads) = merge(default, NamedTuple(kws))

    @assert ndim == 3

    β = invtemp(; kws..., T)
    Δmin, Δmax = extrema(series_Δ)
    series_Δ = prec.([Δmin, zero(eltype(series_Δ)), Δmax])

    cmap = Δ -> Δ >= zero(Δ) ? (1 + Δ/Δmax)/2 : (1 - Δ/Δmin)/2 # keep Δ=0 at center of colorscheme and linear to limits
    colors = cgrad(colormap)

    kp = kpath(; sgnum, kws...)
    ibz = polyhedron(vrep(unique(values(kp.points))))
    klat = Iterators.product(ntuple(n -> AutoSymPTR.ptrpoints(prec, N_k), ndim)...)
    Kmask = in.(collect.(klat), Ref(ibz))
    limits = map(map(i -> extrema(p[i] for p in points(ibz)), ntuple(identity,ndim))) do (a, b)
        len = b - a
        (a-len/10, b+len/10)
    end

    fig = Figure(resolution=(1600,800))

    i = 0
    for (; Ω, vcomp, plot_ibz) in config_vcomp
        plot_ibz || continue
        i += 1
        data_σ_k = map(series_Δ) do Δ
            μ, = findchempot(; T, kws..., Δ)
            σ_k, info_k = conductivity_solver(; kws..., T, μ, Δ, vcomp,
                                        bandwidth_bound=Ω, choose_kω_order=(alg...)->true,
                                        lims_Σ=fermi_window_limits(Ω, β))
            w = σ_k.f.w
            hv_k = AutoBZCore.FourierPTR(w, prec, Val(ndims(w.series)), N_k).s
            p_k = [AutoBZCore.MixedParameters(; hv_k=FourierValue(k, hv), Ω=prec(Ω),
                        getfield(σ_k.f.f.p, :kwargs)...) for (hv,k) in zip(hv_k,klat)]
            id_k = string((; info_k..., quad_σ_k=nothing, p=hash(p_k)))

            cache_path = joinpath(cache_dir, cache_file_values_cond_k_cfs)
            # there is no api to access the inner integral so this is inherently unstable
            # would need a nested integrand type to accomplish that
            dat = cache_batchsolve(σ_k.f.f.f.solver, p_k, cache_path, id_k, nthreads)
            σ_max = maximum(σ -> maximum(real(diag(σ))), dat)
            [ (y = real(diag(σ))/σ_max; keep ? y : zero(y)) for (keep,σ) in zip(Kmask,dat) ]
        end

        for pol in 1:length(first(first(data_σ_k)))
            ax = Axis3(fig[i,pol]; azimuth=1.4*pi, elevation=pi/8, limits)
            hidedecorations!(ax)
            hidespines!(ax)
            plot!(ax, kp)
            # mesh!(ax, Mesh(ibz), alpha=0.2)
            for (Δ, data_σ_k_Δ) in zip(series_Δ, data_σ_k)
                mycolor = colors[cmap(Δ)]
                mymap = Colors.alphacolor.(Makie.to_colormap(cgrad([mycolor, mycolor], 100)), alpharamp)
                mymap[1] = RGBAf(0,0,0,0)
                volume!(ax, klat.iterators..., getindex.(data_σ_k_Δ, pol); algorithm=:mip, colormap=mymap)
            end
        end

        Legend(fig[i,end+1],
            map(Δ -> PolyElement(color=colors[cmap(Δ)], strokecolor=:transparent), series_Δ),
            map(Δ -> "Δ=$Δ", series_Δ),
            label=string(vcomp)
        )
    end

    return fig
end
