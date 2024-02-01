using AutoBZ
using AutoBZ: AbstractHamiltonianInterp, SVector
using Brillouin
using Makie
using LinearAlgebra: dot

@recipe(KPathInterpPlot) do scene
    Theme(
        seriescolor=:cyclic_grey_15_85_c0_n256,
        densitycolormap=:viridis,
        alpha=1.0,
    )
end

struct KPathSegment{B,K,L,S}
    basis::B
    kpath::K
    label::L
    setting::S
end

function Makie.convert_arguments(::Type{<:KPathInterpPlot}, kps::KPathSegment, h::AbstractHamiltonianInterp)
    kps.basis
    B = reduce(hcat, collect(kps.basis.vs))
    kloc = cumdists([kps.setting == Brillouin.CARTESIAN ? k : B*k for k in kps.kpath])
    data = stack([gauge(h) isa Hamiltonian ? h(k).values : AutoBZ.to_gauge(Hamiltonian(), h(k)).values for k in kps.kpath])
    return (kloc, data/oneunit(eltype(data)))
end

function Makie.plot!(kpp::KPathInterpPlot{<:Tuple{AbstractVector{<:Real},AbstractMatrix{<:Real}}})
    kloc = kpp[1][]
    data = kpp[2][]
    # TODO make this function observable compatible
    series!(kpp, kloc, data, color=kpp.seriescolor)

    return kpp
end

function Makie.convert_arguments(::Type{<:KPathInterpPlot}, kps::KPathSegment, hv::AbstractVelocityInterp)
    @assert coord(hv) isa Cartesian
    B = reduce(hcat, kps.basis)
    kcart = [kps.setting == Brillouin.CARTESIAN ? k : B*k for k in kps.kpath]
    kloc = cumdists(kcart)
    data = [gauge(hv) isa Hamiltonian ? hv(k) : ((hk, vk) = hv(k); (hkh, vkh) = AutoBZ.to_gauge(Hamiltonian(), hk, vk.data); (hkh, SVector(vkh))) for k in kps.kpath]
    y = similar(kloc,  length(data[1][1].values), length(kloc))
    u = diff(kloc)
    uvec = diff(kcart)
    v = similar(u, length(data[1][1].values), length(u))
    for (i,(hk, vk)) in enumerate(data)
        y[:,i] .= hk.values

        i > length(uvec) && break
        v[:,i] .= (dot(real.(getindex.(vk,n,n)), uvec[i]) for n in 1:length(hk.values))
    end
    return (kloc, y, u, v)
end

function Makie.plot!(kpp::KPathInterpPlot{<:Tuple{AbstractVector{<:Real},AbstractMatrix{<:Real},AbstractVector{<:Real},AbstractMatrix{<:Real}}})
    x  = kpp[1][]
    ys = kpp[2][]
    u  = kpp[3][]
    vs = kpp[4][]
    # TODO make this function observable compatible
    for (y, v) in zip(eachrow(ys), eachrow(vs))
        lines!(kpp, x, y)
        arrows!(kpp, x[begin:end-1], y[begin:end-1], u, v)
    end
    return kpp
end

function Makie.convert_arguments(::Type{<:KPathInterpPlot}, kps::KPathSegment, freq::AbstractVector{<:Number}, density::FourierIntegrand)
    B = reduce(hcat, kps.basis)
    kloc = cumdists([kps.setting == Brillouin.CARTESIAN ? k : B*k for k in kps.kpath])
    data = [density(FourierValue(k, density.w(k)), f) for k in kps.kpath, f in freq]
    return (kloc, freq/oneunit(eltype(freq)), data/maximum(data))
end

function Makie.plot!(kpp::KPathInterpPlot{<:Tuple{AbstractVector{<:Real},AbstractVector{<:Real},AbstractMatrix{<:Real}}})
    kloc = kpp[1][]
    freq = kpp[2][]
    data = kpp[3][]
    # TODO make this function observable compatible
    heatmap!(kpp, kloc, freq, data; colormap=kpp.densitycolormap, alpha=kpp.alpha)

    return kpp
end

function bandplot!(fig, kpi::KPathInterpolant, args...; kws...)
    len = 0.0
    reuse = (fig.layout.size[1] == 1) && (fig.layout.size[2] >= length(kpi.kpaths)) && !isempty(fig.layout.content) # plot into existing axes
    dat = map(enumerate(zip(kpi.kpaths, kpi.labels))) do (i, (kpath, label))
        B = reduce(hcat, kpi.basis)
        local_xs = cumdists([kpi.setting == Brillouin.CARTESIAN ? k : B*k for k in kpath])
        len += local_xs[end]-local_xs[begin]
        kps = KPathSegment(kpi.basis, kpath, label, kpi.setting)
        axs = reuse ? fig.layout.content[i].content : Axis(fig[1,i], xticks = ([local_xs[i] for i in keys(kps.label)], map(string, values(kps.label))),
            limits=(extrema(local_xs),nothing))
        return (axs, kps)
    end
    for ((ax1,_), (ax2,_)) in zip(@view(dat[begin:end-1]),@view(dat[begin+1:end]))
        linkyaxes!(ax1, ax2)
        hideydecorations!(ax2, ticks = false, grid = false)
    end
    for (ax, kps) in dat
        kpathinterpplot!(ax, kps, args...; kws...)
    end
    for (i,(_, kps)) in enumerate(dat)
        B = reduce(hcat, kpi.basis)
        local_xs = cumdists([kpi.setting == Brillouin.CARTESIAN ? k : B*k for k in kps.kpath])

        colsize!(fig.layout, i, Relative((local_xs[end]-local_xs[begin])/len))
    end
    return dat
end

function bandplot(args...; kws...)
    fig = Figure()
    bandplot!(fig, args...; kws...)
    return fig
end

function inset_axis!(fig, ax; z = 300, extent = (0.25, 0.75, 0.25, 0.75), axis_kwargs...)

    bbox = lift(ax.scene.px_area) do pxa
        _l, _b = minimum(pxa)
        _r, _t = maximum(pxa)
        l = _l + extent[1] * (_r - _l)
        r = _l + extent[2] * (_r - _l)
        b = _b + extent[3] * (_t - _b)
        t = _b + extent[4] * (_t - _b)
        BBox(l, r, b, t)
    end

    inset_ax = Axis(fig, bbox=bbox; axis_kwargs...)

    translate_forward!(x::Makie.Transformable, z) = translate!(Accum, x, 0, 0, z)
    translate_forward!(ax::MakieLayout.LineAxis, z) = foreach(x -> translate_forward!(x, z), values(ax.elements))
    translate_forward!(_) = nothing

    foreach(x -> translate_forward!(x, z), values(inset_ax.elements))
    translate!(Accum, inset_ax.scene, 0, 0, z)

    return inset_ax
end
