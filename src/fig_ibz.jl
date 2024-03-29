using GLMakie
using Brillouin
using SymmetryReduceBZ
using Polyhedra: Polyhedron, Mesh, points, polyhedron, vrep
using FourierSeriesEvaluators: workspace_evaluate!, workspace_contract!

function fig_ibz(; kpath=sgnum_path, colormap=:Spectral, N_k=30, sgnum=76, whichsym=13, pol=1,
    ibzcolor = :green, center = false, center_half = 0, config_ibz,
    alpharamp = range(0,1,length=256), cache_file_values_cond_k_cfs="cache-values-cond-k-cfs.jld2", kws...)
    (; T, chempot, scalar_func, vcomp, ndim, prec, cache_dir, nthreads) = merge(default, NamedTuple(kws))

    @assert ndim == 3

    β = invtemp(; kws..., T)

    kp = kpath(; sgnum, kws...)
    ibz = polyhedron(vrep(unique(values(kp.points))))
    klat = Iterators.product(ntuple(n -> AutoSymPTR.ptrpoints(prec, N_k), ndim)...)
    Kmask = fill(true, size(klat)) # in.(collect.(klat), Ref(ibz))
    limits = map(map(i -> extrema(p[i] for p in points(ibz)), ntuple(identity,ndim))) do (a, b)
        len = b - a
        (a-len/10, b+len/10)
    end
    limits = center ? ntuple(_ -> (-0.5, 0.5), 3) : ntuple(_ -> (0.0, 1.0), 3)

    fig = Figure(resolution=(1600,800))
    mymap = Colors.alphacolor.(Makie.to_colormap(cgrad([ibzcolor, ibzcolor], 100)), alpharamp)
    mymap[1] = RGBAf(0,0,0,0)
    i = 0
    for (; Ω, vcomp, label, plot_kws) in config_ibz
        i += 1
        μ, = chempot(; T, kws...)
        σ_k, info_k = conductivity_solver(; kws..., T, μ, vcomp,
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
        σ_max = maximum(σ -> maximum(scalar_func(σ)), dat)
        data_σ_k = [ (y = scalar_func(σ)/σ_max; keep ? y : zero(y)) for (keep,σ) in zip(Kmask,dat) ]

        ax = Axis3(fig[1,i]; azimuth=2pi/3, elevation=-pi/3, limits)
        hidedecorations!(ax)
        hidespines!(ax)
        plot!(ax, kp)
        # mesh!(ax, Mesh(ibz), alpha=0.2)
        contour!(ax, (center ? ntuple(n -> n == center_half ? klat.iterators[n][(end-div(size(data_σ_k,n),2)+1):end] .- 0.5 : klat.iterators[n] .- 0.5, 3) : klat.iterators)..., center ? circshift(data_σ_k, map(l -> div(l,2), size(data_σ_k)))[ntuple(n -> n == center_half ? (lastindex(data_σ_k,n)-div(size(data_σ_k,n),2)+1:(lastindex(data_σ_k,n))) : (firstindex(data_σ_k,n):lastindex(data_σ_k,n)), 3)...] : data_σ_k; plot_kws...)
        Label(fig[2,i], label)

    end
    for j in 1:i
        colsize!(fig.layout, j, Relative(1/i))
    end
    rowsize!(fig.layout, 1, Relative(6/7))
    rowsize!(fig.layout, 2, Relative(1/7))


    return fig
end
