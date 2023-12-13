using CairoMakie
using AutoBZ
using AutoBZ: SVector
using LinearAlgebra
using Brillouin
using ColorSchemes, Colors


function fig3a(; cintra=:orange, cinter=:green, alpha=1.0, ylims=(0, 50), kws...)

    (; scalarize, scalarize_text, Nk, Nω, NΩ, σatol, σrtol, t, Ωintra, Ωinter, Ωlims, μlims, prec, σfalg, gauge, coord, σudisplay, σufactor, nsp, ndim) = merge(default, NamedTuple(kws))
    h, bz = t2gmodel(; kws..., gauge=Wannier())

    μ = findchempot(; kws...)
    shift!(h, μ)
    hvintra = GradientVelocityInterp(h, bz.A; coord, vcomp=Intra(), gauge)
    hvinter = GradientVelocityInterp(h, bz.A; coord, vcomp=Inter(), gauge)

    η = fermi_liquid_scattering(; kws...)
    β = fermi_liquid_beta(; kws...)
    Σ = EtaSelfEnergy(η)
    μintra = -Ωintra/2
    μinter = -Ωinter/2
    ocintra = FourierIntegrand((x, f) -> ustrip(scalarize(AutoBZ.transport_fermi_integrand_inside(f*unit(t); Σ, n=0, β, Ω=prec(Ωintra), μ=prec(μintra), hv_k=x))), hvintra)
    ocinter = FourierIntegrand((x, f) -> ustrip(scalarize(AutoBZ.transport_fermi_integrand_inside(f*unit(t); Σ, n=0, β, Ω=prec(Ωinter), μ=prec(μinter), hv_k=x))), hvinter)

    oc_kintegrand_intra = OpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), σfalg, hvintra; Σ, β, Ω=prec(Ωintra), μ=prec(μintra), reltol=σrtol)
    oc_kintegrand_inter = OpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), σfalg, hvinter; Σ, β, Ω=prec(Ωinter), μ=prec(μinter), reltol=σrtol)

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

    alpharamp = range(0, 1, length=256) # log.(10, 9*range(0, 1, length=256) .+ 1)
    cintragrad = Colors.alphacolor.(Makie.to_colormap(cgrad([cintra, cintra], 100)), alpharamp)
    cintergrad = Colors.alphacolor.(Makie.to_colormap(cgrad([cinter, cinter], 100)), alpharamp)

    fig = Figure(resolution=(800,1600))

    condlayout = GridLayout()
    Ωs = range(prec.(Ωlims)..., length=NΩ)
    ax = Axis(fig, title="Orbital structure of optical conductivity",
        xlabel="Ω ($(unit(eltype(Ωs))))", xticks=ustrip.(unique(append!(collect(Ωlims), [Ωinter, Ωintra]))),
        ylabel="$(scalarize_text) ($(σudisplay))",
        limits=(ustrip.(Ωlims), ylims))
    condlayout[1,1] = ax

    cond = interpolateconductivity(; kws..., vcomp=Whole())
    condintra = interpolateconductivity(; kws..., vcomp=Intra())
    condinter = interpolateconductivity(; kws..., vcomp=Inter())

    lines!(ax, ustrip.(Ωs), ustrip.(uconvert.(σudisplay, (nsp*σufactor/(2pi)^ndim) .* scalarize.(cond.(Ωs)))), color=:black, label="whole")
    lines!(ax, ustrip.(Ωs), ustrip.(uconvert.(σudisplay, (nsp*σufactor/(2pi)^ndim) .* scalarize.(condintra.(Ωs)))), color=cintra, label="inter-band")
    lines!(ax, ustrip.(Ωs), ustrip.(uconvert.(σudisplay, (nsp*σufactor/(2pi)^ndim) .* scalarize.(condinter.(Ωs)))), color=cinter, label="inter-band")
    axislegend(ax)

    kps = KPathSegment(kpi.basis, kpi.kpaths[1], kpi.labels[1], kpi.setting)
    kloc = cumdists([ustrip.(bz.B) * k for k in kps.kpath])
    ωlims = μlims ./ 2 .- μ
    freqs = range(prec.(ustrip.(ωlims))..., length=Nω)

    structurelayout = GridLayout(nrow=4, ncol=2, parent=fig)

    axintra = Axis(fig,
    xlabel="Spectral density of $(scalarize_text)", ylabel="ω (eV)",
    xticks=(kloc[collect(keys(kps.label))], map(string, values(kps.label))),
    limits = (extrema(kloc), extrema(freqs))
    )
    structurelayout[2,1] = axintra


    axinter = Axis(fig,
    xlabel="Spectral density of $(scalarize_text)", ylabel="ω (eV)",
    xticks=(kloc[collect(keys(kps.label))], map(string, values(kps.label))),
    limits = (extrema(kloc), extrema(freqs))
    )
    structurelayout[4,1] = axinter


    kpathinterpplot!(axintra, kps, freqs, ocintra; alpha, densitycolormap=cintragrad)
    kpathinterpplot!(axintra, kps, h)

    kpathinterpplot!(axinter, kps, freqs, ocinter; alpha, densitycolormap=cintergrad)
    kpathinterpplot!(axinter, kps, h)


    insetintra = inset_axis!(fig, axintra; extent = (0.15, 0.45, 0.45, 0.75),
        limits=((1.0,2.5), (-0.05, 0.05)), xticklabelsvisible=false, xticksvisible=false, xgridvisible=false, ygridvisible=false)

    kpathinterpplot!(insetintra, kps, freqs, ocintra; alpha, densitycolormap=cintragrad) #densitycolormap=Makie.Reverse(:RdBu))
    kpathinterpplot!(insetintra, kps, h)

    insetinter = inset_axis!(fig, axinter; extent = (0.15, 0.45, 0.45, 0.75),
        limits=((1.0,2.0), (-0.2, 0.05)), xticklabelsvisible=false, xticksvisible=false, xgridvisible=false, ygridvisible=false)

    kpathinterpplot!(insetinter, kps, freqs, ocinter; alpha, densitycolormap=cintergrad) #densitycolormap=Makie.Reverse(:RdBu))
    kpathinterpplot!(insetinter, kps, h)

    axkintra = Axis(fig, ylabel="∫ $(scalarize_text)(k,ω) dω", limits=(extrema(kloc), nothing), ygridvisible=false, yticks=[0.0])
    structurelayout[1,1] = axkintra
    linkxaxes!(axkintra, axintra)
    hidexdecorations!(axkintra, ticks = false, grid = false)


    axkinter = Axis(fig, ylabel="∫ $(scalarize_text)(k,ω) dω", limits=(extrema(kloc), nothing), ygridvisible=false, yticks=[0.0])
    structurelayout[3,1] = axkinter
    linkxaxes!(axkinter, axinter)
    hidexdecorations!(axkinter, ticks = false, grid = false)

    # problem with nearly-degenerate eigenvectors is that
    dat_k_intra = [oc_kintegrand_intra(map(prec, k), AutoBZCore.NullParameters()) for k in kpi]
    dat_k_inter = [oc_kintegrand_inter(map(prec, k), AutoBZCore.NullParameters()) for k in kpi]


    lines!(axkintra, kloc, ustrip.(scalarize.(dat_k_intra)), color=cintra)
    lines!(axkinter, kloc, ustrip.(scalarize.(dat_k_inter)), color=cinter)

    axfintra = Axis(fig, xlabel="∫ $(scalarize_text)(k,ω) dk", xgridvisible=false, xticks=[0.0])
    structurelayout[2,2] = axfintra
    linkyaxes!(axfintra, axintra)
    hideydecorations!(axfintra, ticks = false, grid = false)

    axfinter = Axis(fig, xlabel="∫ $(scalarize_text)(k,ω) dk", xgridvisible=false, xticks=[0.0])
    structurelayout[4,2] = axfinter
    linkyaxes!(axfinter, axinter)
    hideydecorations!(axfinter, ticks = false, grid = false)

    oc_fintegrand_intra = interpolateconductivityk(; μoffset=μintra, ωlims, kws..., vcomp=Intra(), Ω=Ωintra)
    oc_fintegrand_inter = interpolateconductivityk(; μoffset=μinter, ωlims, kws..., vcomp=Inter(), Ω=Ωinter)

    # dat_f_intra = [oc_fintegrand_intra(f*unit(t), AutoBZCore.NullParameters()) for f in freqs]
    # dat_f_inter = [oc_fintegrand_inter(f*unit(t), AutoBZCore.NullParameters()) for f in freqs]

    lines!(axfintra, ustrip.(uconvert.(unit(σatol)/unit(t), scalarize.(oc_fintegrand_intra.(unit(t) .* freqs)))), freqs, color=cintra)
    lines!(axfinter, ustrip.(uconvert.(unit(σatol)/unit(t), scalarize.(oc_fintegrand_inter.(unit(t) .* freqs)))), freqs, color=cinter)


    colsize!(structurelayout, 1, Relative(4/5))
    colsize!(structurelayout, 2, Relative(1/5))
    rowsize!(structurelayout, 1, Relative(1/5/2))
    rowsize!(structurelayout, 2, Relative(4/5/2))
    rowsize!(structurelayout, 3, Relative(1/5/2))
    rowsize!(structurelayout, 4, Relative(4/5/2))

    fig.layout[1,1] = condlayout
    fig.layout[2,1] = structurelayout
    rowsize!(fig.layout, 1, Relative(2/9))
    rowsize!(fig.layout, 2, Relative(7/9))


    Legend(structurelayout[1,2],
        [LineElement(color=:black), LineElement(color=cintra)],
        ["spectrum", "intra-band"],
    )

    Legend(structurelayout[3,2],
        [LineElement(color=:black), LineElement(color=cinter)],
        ["spectrum", "inter-band"],
    )

    return fig

end
