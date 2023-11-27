using CairoMakie
using AutoBZ
using AutoBZ: SVector
using LinearAlgebra
using Brillouin

function fig3a(scalarize=real∘tr)
    H = model()
    bz = load_bz(CubicSymIBZ(), one(SMatrix{3,3,Float64,9}))

    hvinter = GradientVelocityInterp(H, bz.A; coord=Cartesian(), vcomp=Inter())
    hvintra = GradientVelocityInterp(H, bz.A; coord=Cartesian(), vcomp=Intra())

    pts = Dict{Symbol,SVector{3,Float64}}(
        :R => [0.5, 0.5, 0.5],
        :M => [0.5, 0.5, 0.0],
        :Γ => [0.0, 0.0, 0.0],
        :X => [0.0, 0.5, 0.0],
    )
    paths = [
        [:Γ, :R, :X, :M, :Γ],
    ]
    basis = Brillouin.KPaths.reciprocalbasis(collect(eachcol(bz.A)))
    setting = Ref(Brillouin.LATTICE)
    kp = KPath(pts, paths, basis, setting)
    kpi = interpolate(kp, 1000)

    fig = Figure()

    kps = AutoBZBrillouinMakieViz.KPathSegment(kpi.basis, kpi.kpaths[1], kpi.labels[1], kpi.setting)
    kloc = cumdists([bz.B * k for k in kps.kpath])
    freqs = range(-5, 5, length=1000)

    ax = Axis(fig[2,1],
        xlabel="Spectral density of tr(σ)", ylabel="ω (eV)",
        xticks=(kloc[collect(keys(kps.label))], map(string, values(kps.label))),
        limits = (extrema(kloc), extrema(freqs))
    )

    axk = Axis(fig[1,1], ylabel="∫ tr(σ(k,ω)) dω", limits=(extrema(kloc), nothing))
    linkxaxes!(axk, ax)
    hidexdecorations!(axk, ticks = false, grid = false)

    lines!(axk, kloc, dat_k_intra, color=cintra)
    lines!(axk, kloc, dat_k_inter, color=cinter)

    # dat_f_intra = [real(oc_fintegrand_intra(f, AutoBZCore.NullParameters())[1,1]) for f in freqs]
    # dat_f_inter = [real(oc_fintegrand_inter(f, AutoBZCore.NullParameters())[1,1]) for f in freqs]


    # dat_f_intra = [real(tr(oc_fintegrand_intra(f, AutoBZCore.NullParameters()))) for f in freqs]
    # dat_f_inter = [real(tr(oc_fintegrand_inter(f, AutoBZCore.NullParameters()))) for f in freqs]

    axf = Axis(fig[2,2], xlabel="∫ tr(σ(k,ω)) dk", limits=(nothing, extrema(freqs)))
    linkyaxes!(axf, ax)
    hideydecorations!(axf, ticks = false, grid = false)

    # lines!(axf, dat_f_intra, freqs, color=cintra)
    # lines!(axf, dat_f_inter, freqs, color=cinter)


    Legend(fig[1,2],
        [LineElement(color=:black), LineElement(color=cintra), LineElement(color=cinter)],
        ["spectrum", "intra-band", "inter-band"],
    )

    colsize!(fig.layout, 1, Relative(0.8))
    colsize!(fig.layout, 2, Relative(0.2))
    rowsize!(fig.layout, 1, Relative(0.2))
    rowsize!(fig.layout, 2, Relative(0.8))

end
