using CairoMakie
using AutoBZ
using AutoBZ: SVector
using LinearAlgebra
using Brillouin
using ColorSchemes, Colors


function fig3a(; scalarize=real∘tr, scalarize_text="Tr σ",  t=-0.25u"eV", t′=0.05u"eV", Δ=0.0u"eV",
    Ωintra = 0.0u"eV", Ωinter = 0.4u"eV", T = 100.0u"K", T₀=300.0u"K", Z=0.5, atol=1e-2, rtol=1e-4, ν=1.0, nsp=2, falg=AuxQuadGKJL(), kalg=PTR(npt=100))

    bz = load_bz(CubicSymIBZ(), one(SMatrix{3,3,Float64,9}) * u"Å")
    n = interpolatedensity(t=t, t′=t′, Δ=Δ, T=T, T₀=T₀, Z=Z, atol=atol, rtol=rtol)

    H = t2gmodel(t=t, t′=t′, Δ=Δ, gauge=Wannier())
    chempot = uconvert(unit(t), findchempot(n; μlims=(n.lb[1], n.ub[1]), ν=ν, nsp=nsp))
    shift!(H, chempot)
    hvintra = GradientVelocityInterp(H, bz.A; coord=Cartesian(), vcomp=Intra())
    hvinter = GradientVelocityInterp(H, bz.A; coord=Cartesian(), vcomp=Inter())

    Ωintra = uconvert(unit(t), Ωintra)
    Ωinter = uconvert(unit(t), Ωinter)
    η = uconvert(unit(t), T[1]^2*u"k_au"*pi/(Z*T₀))
    Σ = ConstScalarSelfEnergy(-im*η)
    β = 1/uconvert(unit(t), u"k_au"*T[1])
    μintra = -Ωintra/2
    μinter = -Ωinter/2
    ocintra = FourierIntegrand((x, f) -> ustrip(scalarize(AutoBZ.transport_fermi_integrand_inside(f*oneunit(t), Σ, nothing, 0, β, Ωintra, μintra, x))), hvintra)
    ocinter = FourierIntegrand((x, f) -> ustrip(scalarize(AutoBZ.transport_fermi_integrand_inside(f*oneunit(t), Σ, nothing, 0, β, Ωinter, μinter, x))), hvinter)

    oc_kintegrand_intra = OpticalConductivityIntegrand(AutoBZ.fermi_window_limits(Ωintra, β)..., falg, hvintra, Σ, β, Ωintra, μintra, reltol=rtol)
    oc_kintegrand_inter = OpticalConductivityIntegrand(AutoBZ.fermi_window_limits(Ωinter, β)..., falg, hvinter, Σ, β, Ωinter, μinter, reltol=rtol)

    oc_fintegrand_intra = OpticalConductivityIntegrand(bz, kalg, hvintra, Σ, β, Ωintra, μintra)
    oc_fintegrand_inter = OpticalConductivityIntegrand(bz, kalg, hvinter, Σ, β, Ωinter, μinter)

    pts = Dict{Symbol,SVector{3,Float64}}(
        :R => [0.5, 0.5, 0.5],
        :M => [0.5, 0.5, 0.0],
        :Γ => [0.0, 0.0, 0.0],
        :X => [0.0, 0.5, 0.0],
    )
    paths = [
        [:Γ, :R, :X, :M, :Γ],
    ]
    basis = Brillouin.KPaths.reciprocalbasis(collect(eachcol(bz.A*u"Å^-1")))
    setting = Ref(Brillouin.LATTICE)
    kp = KPath(pts, paths, basis, setting)
    kpi = interpolate(kp, 1000)

    alpharamp = range(0, 1, length=256) .^ 2 # log.(10, 9*range(0, 1, length=256) .+ 1)
    cintra = :orange
    cintragrad = Colors.alphacolor.(Makie.to_colormap(cgrad([cintra, cintra], 100)), alpharamp)
    cinter = :green
    cintergrad = Colors.alphacolor.(Makie.to_colormap(cgrad([cinter, cinter, cinter], 100)), sqrt.(alpharamp))

    fig = Figure(resolution=(800,1200))

    kps = KPathSegment(kpi.basis, kpi.kpaths[1], kpi.labels[1], kpi.setting)
    kloc = cumdists([bz.B*u"Å" * k for k in kps.kpath])
    freqs = range(sort(((-4t - chempot)/oneunit(t), (4t - chempot)/oneunit(t)))..., length=1000)


    axintra = Axis(fig[2,1],
    xlabel="Spectral density of $(scalarize_text)", ylabel="ω (eV)",
    xticks=(kloc[collect(keys(kps.label))], map(string, values(kps.label))),
    limits = (extrema(kloc), extrema(freqs))
    )

    axinter = Axis(fig[4,1],
    xlabel="Spectral density of $(scalarize_text)", ylabel="ω (eV)",
    xticks=(kloc[collect(keys(kps.label))], map(string, values(kps.label))),
    limits = (extrema(kloc), extrema(freqs))
    )

    kpathinterpplot!(axintra, kps, freqs, ocintra, alpha=1.0, densitycolormap=cintragrad) #densitycolormap=Makie.Reverse(:RdBu))
    kpathinterpplot!(axintra, kps, H)

    kpathinterpplot!(axinter, kps, freqs, ocinter, alpha=0.5, densitycolormap=cintergrad) #densitycolormap=Makie.Reverse(:RdBu))
    kpathinterpplot!(axinter, kps, H)


    insetintra = inset_axis!(fig, axintra; extent = (0.15, 0.45, 0.45, 0.75), limits=((1.0,2.5), (-0.1, 0.1)), xgridvisible=false, ygridvisible=false)

    kpathinterpplot!(insetintra, kps, freqs, ocintra, alpha=1.0, densitycolormap=cintragrad) #densitycolormap=Makie.Reverse(:RdBu))
    kpathinterpplot!(insetintra, kps, H)

    insetinter = inset_axis!(fig, axinter; extent = (0.15, 0.45, 0.45, 0.75), limits=((1.0,2.0), (-0.2, 0.05)), xgridvisible=false, ygridvisible=false)

    kpathinterpplot!(insetinter, kps, freqs, ocinter, alpha=1.0, densitycolormap=cintergrad) #densitycolormap=Makie.Reverse(:RdBu))
    kpathinterpplot!(insetinter, kps, H)

    axkintra = Axis(fig[1,1], ylabel="∫ $(scalarize_text)(k,ω) dω", limits=(extrema(kloc), nothing), ygridvisible=false)
    linkxaxes!(axkintra, axintra)
    hidexdecorations!(axkintra, ticks = false, grid = false)


    axkinter = Axis(fig[3,1], ylabel="∫ $(scalarize_text)(k,ω) dω", limits=(extrema(kloc), nothing), ygridvisible=false)
    linkxaxes!(axkinter, axinter)
    hidexdecorations!(axkinter, ticks = false, grid = false)

    # problem with nearly-degenerate eigenvectors is that
    dat_k_intra = [oc_kintegrand_intra(k, AutoBZCore.NullParameters()) for k in kpi]
    dat_k_inter = [oc_kintegrand_inter(k, AutoBZCore.NullParameters()) for k in kpi]


    lines!(axkintra, kloc, ustrip.(scalarize.(dat_k_intra)), color=cintra)
    lines!(axkinter, kloc, ustrip.(scalarize.(dat_k_inter)), color=cinter)

    axfintra = Axis(fig[2,2], xlabel="∫ $(scalarize_text)(k,ω) dk", xgridvisible=false)# limits=(nothing, extrema(freqs)))
    linkyaxes!(axfintra, axintra)
    hideydecorations!(axfintra, ticks = false, grid = false)

    axfinter = Axis(fig[4,2], xlabel="∫ $(scalarize_text)(k,ω) dk", xgridvisible=false)# limits=(nothing, extrema(freqs)))
    linkyaxes!(axfinter, axinter)
    hideydecorations!(axfinter, ticks = false, grid = false)

    # dat_f_intra = [oc_fintegrand_intra(f*unit(t), AutoBZCore.NullParameters()) for f in freqs]
    # dat_f_inter = [oc_fintegrand_inter(f*unit(t), AutoBZCore.NullParameters()) for f in freqs]

    # lines!(axfintra, ustrip.(uconvert.(u"eV*Å^-1", scalarize.(dat_f_intra))), freqs, color=cintra)
    # lines!(axfinter, ustrip.(uconvert.(u"eV*Å^-1", scalarize.(dat_f_inter))), freqs, color=cinter)


    Legend(fig[1,2],
        [LineElement(color=:black), LineElement(color=cintra)],
        ["spectrum", "intra-band"],
    )

    Legend(fig[3,2],
        [LineElement(color=:black), LineElement(color=cinter)],
        ["spectrum", "inter-band"],
    )

    colsize!(fig.layout, 1, Relative(0.8))
    colsize!(fig.layout, 2, Relative(0.2))
    rowsize!(fig.layout, 1, Relative(0.1))
    rowsize!(fig.layout, 2, Relative(0.4))
    rowsize!(fig.layout, 3, Relative(0.1))
    rowsize!(fig.layout, 4, Relative(0.4))

    return fig

end
