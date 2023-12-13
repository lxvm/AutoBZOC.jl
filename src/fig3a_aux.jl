using CairoMakie
using AutoBZ
using AutoBZ: SVector
using LinearAlgebra
using Brillouin
using ColorSchemes, Colors


function fig3a_aux(; densitycolormap=:RdBu, color=:purple, auxfield=:aux, alpha=1.0,
    Ωaux=0.0u"eV", kws...)

    (; scalarize, scalarize_text, Nk, Nω, NΩ, σatol, σrtol, σauxatol, σauxrtol, t, Ωinter, Ωintra, Ωlims, μlims, prec, σfalg, gauge, coord, auxfun, vcomp) = merge(default, NamedTuple(kws))
    h, bz = t2gmodel(; kws..., gauge=Wannier())

    μ = findchempot(; kws...)
    shift!(h, μ)
    hv = GradientVelocityInterp(h, bz.A; coord, vcomp, gauge)

    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    Σ = EtaSelfEnergy(η)
    Ω = Ωaux
    μoffset = -Ω/2
    oc = FourierIntegrand((x, f) -> ustrip(scalarize(getproperty(AutoBZ.aux_transport_fermi_integrand_inside(f*unit(t), auxfun; Σ, n=0, β, Ω=prec(Ω), μ=prec(μoffset), hv_k=x), auxfield))), hv)

    oc_kintegrand = AuxOpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), σfalg, hv, auxfun; Σ, β, Ω=prec(Ω), μ=prec(μoffset), reltol=AuxValue(σrtol, σauxrtol))

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
    kpi = interpolate(kp, Nk)

    fig = Figure(resolution=(800,1200))

    condlayout = GridLayout()
    Ωs = range(prec.(Ωlims)..., length=NΩ)
    ax = Axis(fig, title="Orbital structure of auxiliary conductivity",
        xlabel="Ω ($(unit(eltype(Ωs))))", xticks=ustrip.(unique(append!(collect(Ωlims), [Ωinter, Ωintra]))),
        ylabel="$(scalarize_text) ($(unit(σauxatol)))",
        limits=(ustrip.(Ωlims), nothing))
    condlayout[1,1] = ax

    cond = interpolateauxconductivity(; kws..., vcomp)
    lines!(ax, ustrip.(Ωs), ustrip.(scalarize.(getproperty.(cond.(Ωs), Ref(auxfield)))); color, label="whole")
    axislegend(ax)

    kps = KPathSegment(kpi.basis, kpi.kpaths[1], kpi.labels[1], kpi.setting)
    kloc = cumdists([bz.B*u"Å" * k for k in kps.kpath])
    ωlims = μlims ./ 2 .- μ
    freqs = range(prec.(ustrip.(ωlims))..., length=Nω)

    structurelayout = GridLayout(nrow=4, ncol=2, parent=fig)

    ax = Axis(fig,
    xlabel="Spectral density of $(scalarize_text)", ylabel="ω (eV)",
    xticks=(kloc[collect(keys(kps.label))], map(string, values(kps.label))),
    limits = (extrema(kloc), extrema(freqs))
    )
    structurelayout[2,1] = ax


    kpathinterpplot!(ax, kps, freqs, oc; alpha, densitycolormap)
    kpathinterpplot!(ax, kps, h)

    inset = inset_axis!(fig, ax; extent = (0.15, 0.45, 0.45, 0.75),
        limits=((1.0,2.0), (-0.2, 0.05)), xticklabelsvisible=false, xticksvisible=false, xgridvisible=false, ygridvisible=false)

    kpathinterpplot!(inset, kps, freqs, oc; alpha, densitycolormap)
    kpathinterpplot!(inset, kps, h)

    axk = Axis(fig, ylabel="∫ $(scalarize_text)(k,ω) dω", limits=(extrema(kloc), nothing), ygridvisible=false, yticks=[0.0])
    structurelayout[1,1] = axk
    linkxaxes!(axk, ax)
    hidexdecorations!(axk, ticks = false, grid = false)

    # problem with nearly-degenerate eigenvectors is that
    dat_k = [oc_kintegrand(map(prec, k), AutoBZCore.NullParameters()) for k in kpi]


    lines!(axk, kloc, ustrip.(scalarize.(getproperty.(dat_k, Ref(auxfield)))); color)

    axf = Axis(fig, xlabel="∫ $(scalarize_text)(k,ω) dk", xgridvisible=false, xticks=[0.0])# limits=(nothing, extrema(freqs)))
    structurelayout[2,2] = axf
    linkyaxes!(axf, ax)
    hideydecorations!(axf, ticks = false, grid = false)

    oc_fintegrand = interpolateauxconductivityk(; μoffset, ωlims, kws..., Ω)

    # dat_f = [oc_fintegrand(f*unit(t), AutoBZCore.NullParameters()) for f in freqs]

    lines!(axf, ustrip.(uconvert.(unit(σauxatol)/unit(t), scalarize.(getproperty.(oc_fintegrand.(unit(t) .* freqs), Ref(auxfield))))), freqs; color)


    colsize!(structurelayout, 1, Relative(4/5))
    colsize!(structurelayout, 2, Relative(1/5))
    rowsize!(structurelayout, 1, Relative(1/5))
    rowsize!(structurelayout, 2, Relative(4/5))

    fig.layout[1,1] = condlayout
    fig.layout[2,1] = structurelayout
    rowsize!(fig.layout, 1, Relative(1/2))
    rowsize!(fig.layout, 2, Relative(1/2))


    Legend(structurelayout[1,2],
        [LineElement(color=:black), LineElement(; color)],
        ["spectrum", "$vcomp"],
    )

    return fig

end
