using GLMakie
using Brillouin
using SymmetryReduceBZ
using Polyhedra: Polyhedron, Mesh, points
using FourierSeriesEvaluators: workspace_evaluate!, workspace_contract!

function fig_ibz(; Ω, colormap=:Spectral, Nk=30, sgnum=76, whichsym=13, kws...)

    (; Δseries, σudisplay, σufactor, σfalg, σatol, σrtol, coord, vcomp, gauge, ndim, prec) = merge(default, NamedTuple(kws))

    Δmax = maximum(Δseries)
    Δzero = zero(eltype(Δseries))
    Δmin = minimum(Δseries)

    cmap = Δ -> Δ >= zero(Δ) ? (1 + Δ/Δmax)/2 : (1 - Δ/Δmin)/2 # keep Δ=0 at center of colorscheme and linear to limits
    colors = cgrad(colormap)

    H_k_Δ = map((Δmin, Δzero, Δmax)) do Δ
        evalHVk(; kws..., Δ, Nk)
    end

    h, bz = t2gmodel(; kws..., gauge=Wannier(), Δ=Δmax)
    @assert bz.B ≈ Diagonal(bz.B) "we require orthogonal basis vectors for plotting"
    atom_species = [
        "Sr",
        "V",
        "O",
        "O",
        "O",
    ]
    atom_pos = [
        0.0 0.0 0.0
        0.5 0.5 0.5
        0.0 0.5 0.5
        0.5 0.0 0.5
        0.5 0.5 0.0
    ]
    ibz = AutoBZ.load_bz(IBZ{3,Polyhedron}(), bz.A, bz.B, atom_species, atom_pos')
    hull_lat = ibz.syms[whichsym] * ibz.lims.p # for the svo crystal use sym 26
    # hull_cart = ibz.B * hull_lat # the hull corresponding to the Brillouin.jl ibz

    # xcart = ntuple(n -> range(extrema(getindex.(points(hull_lat),n))..., length=Nk, step=1//Nk), ndim)
    xcart = ntuple(n -> AutoSymPTR.ptrpoints(prec, Nk), ndim)
    Kmask = in.(collect.(Iterators.product(xcart...)), Ref(hull_lat))


    Σ = EtaSelfEnergy(fermi_liquid_scattering(; kws...))
    β = fermi_liquid_beta(; kws...)
    hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
    oc_integrand = OpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), σfalg, hv; Σ, β, Ω, abstol=σatol/det(bz.B)/nsyms(bz), reltol=σrtol)

    data = map(H_k_Δ) do x
        H_k, μ = x
        [f ? oc_integrand(FourierValue(x,v), AutoBZCore.MixedParameters(; μ)) : zero(SMatrix{ndim,ndim,typeof(σatol/det(bz.B)),ndim^2}) for (x,f,v) in zip(Iterators.product(xcart...),Kmask,H_k)]
    end

    polx = σ -> real(σ[1,1])
    poly = σ -> real(σ[2,2])

    dataxu = map(data) do σ
        polx.(σ)
    end
    maxx = maximum(maximum, dataxu)
    datax = dataxu ./ maxx

    datayu = map(data) do σ
        poly.(σ)
    end
    maxy = maximum(maximum, datayu)
    datay = dataxu ./ maxy


    kp = irrfbz_path(sgnum, eachcol(bz.B'bz.A)) #eachcol(bz.A))
    # pts = Dict{Symbol,SVector{3,Float64}}(
    #     :Z => [0.0, 0.0, 0.5],
    #     :R => [0.5, 0.0, 0.5],
    #     :M => [0.5, 0.5, 0.0],
    #     :A => [0.5, 0.5, 0.5],
    #     :Γ => [0.0, 0.0, 0.0],
    #     :X => [0.5, 0.0, 0.0],
    # )
    # paths = [
    #     [:Γ, :X, :M, :Γ, :Z, :R, :A, :Z],
    #     [:X, :R],
    #     [:M, :A],
    # ]
    # basis0 = Brillouin.KPaths.reciprocalbasis(collect(eachcol(bz.B'bz.A)))
    # setting = Ref(Brillouin.LATTICE)
    # kp = KPath(pts, paths, basis0, setting)

    fig = Figure(resolution=(1600,800))

    limits = map(((a, b),) -> (len = b-a; (a-len/10, b+len/10)), map(extrema, xcart))

    ax1 = Axis3(fig[1,1]; azimuth=1.4*pi, elevation=pi/8, limits)
    hidedecorations!(ax1)
    hidespines!(ax1)
    plot!(ax1, kp)
    # mesh!(ax1, Mesh(hull_lat), alpha=0.2)


    ax2 = Axis3(fig[1,2]; azimuth=1.4*pi, elevation=pi/8, limits)
    hidedecorations!(ax2)
    hidespines!(ax2)
    plot!(ax2, kp)

    alpharamp = range(0,1,length=256) .^ 4 # log.(10, 9*range(0, 1, length=256) .+ 1)

    for (Δ, xΔ, yΔ) in zip(Δseries, datax, datay)
        mycolor = colors[cmap(Δ)]
        mymap = Colors.alphacolor.(Makie.to_colormap(cgrad([mycolor, mycolor], 100)), alpharamp)
        mymap[1] = RGBAf(0,0,0,0)
        volume!(ax1, collect.(xcart)..., xΔ; algorithm=:mip, colormap=mymap)
        volume!(ax2, collect.(xcart)..., yΔ; algorithm=:mip, colormap=mymap)
    end
    Legend(fig[1,3],
        map(Δ -> PolyElement(color=colors[cmap(Δ)], strokecolor=:transparent), [Δmin, Δzero, Δmax]),
        map(Δ -> "Δ=$Δ", [Δmin, Δzero, Δmax]),
    )
    return fig
end

function evalHVk(; kws...)
    (; Nk, coord, vcomp, gauge, nworkers, prec) = merge(default, NamedTuple(kws))
    h, bz = t2gmodel(; kws..., gauge=Wannier())
    hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)
    w = AutoBZCore.workspace_allocate_vec(hv, period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
    H_k = AutoBZCore.FourierPTR(w, prec, Val(ndims(hv)), Nk).s
    μ = findchempot(; kws...)
    return H_k, μ
end
