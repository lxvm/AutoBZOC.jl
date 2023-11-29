using CairoMakie
using AutoBZ
using AutoBZ: SVector
using LinearAlgebra
using Brillouin
using ColorSchemes, Colors


function fig3a(;
    io = stdout,
    verb = true,
    scalarize = default.scalarize,
    scalarize_text = default.scalarize_text,
    t = default.t,
    t′ = default.t′,
    Δ = default.Δ,
    nalg = default.nalg,
    natol = default.natol,
    nrtol = default.nrtol,
    μlims = default.μlims,
    Ωintra = default.Ωintra,
    Ωinter = default.Ωinter,
    T = default.T,
    T₀ = default.T₀,
    Z = default.Z,
    Ωlims = default.Ωlims,
    atol = default.σatol,
    rtol = default.σrtol,
    ν = default.ν,
    nsp = default.nsp,
    falg = default.falg,
    kalg = default.kalg,
    bzkind = default.bzkind,
    cintra = default.cintra,
    cinter = default.cinter,
)

    bz = load_bz(bzkind, one(SMatrix{3,3,Float64,9}) * u"Å")

    H = t2gmodel(t=t, t′=t′, Δ=Δ, gauge=Wannier())
    chempot = uconvert(unit(t), findchempot(; io=io, verb=verb, t=t, t′=t′, Δ=Δ, T=T, T₀=T₀, Z=Z, μlims=μlims, ν=ν, nsp=nsp, alg=nalg, kalg=kalg, falg=falg, atol=natol, rtol=nrtol, bzkind=bzkind))
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

    # oc_fintegrand_intra = OpticalConductivityIntegrand(bz, kalg, hvintra, Σ, β, Ωintra, μintra)
    # oc_fintegrand_inter = OpticalConductivityIntegrand(bz, kalg, hvinter, Σ, β, Ωinter, μinter)

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

    alpharamp = range(0, 1, length=256) # log.(10, 9*range(0, 1, length=256) .+ 1)    
    cintragrad = Colors.alphacolor.(Makie.to_colormap(cgrad([cintra, cintra], 100)), alpharamp)
    cintergrad = Colors.alphacolor.(Makie.to_colormap(cgrad([cinter, cinter], 100)), alpharamp)

    fig = Figure(resolution=(800,1600))

    condlayout = GridLayout()
    Ωs = range(Ωlims..., length=2000)
    ax = Axis(fig, title="Orbital structure of optical conductivity",
        xlabel="Ω ($(unit(eltype(Ωs))))", xticks=ustrip.(unique(append!(collect(Ωlims), [Ωinter, Ωintra]))),
        ylabel="$(scalarize_text) ($(unit(atol)))", yscale=log10,
        limits=(ustrip.(Ωlims), nothing))
    condlayout[1,1] = ax

    cond = interpolateconductivitykw(vcomp=Whole(), t=t, t′=t′, Δ=Δ, T=T, T₀=T₀, Z=Z, kalg=kalg, falg=falg, natol=natol, nrtol=nrtol, nalg=nalg, bzkind=bzkind,
                μlims=μlims, ν=ν, nsp=nsp, Ωlims=Ωlims, atol=atol, rtol=rtol, io=io, verb=verb)

    condintra = interpolateconductivitykw(vcomp=Intra(), t=t, t′=t′, Δ=Δ, T=T, T₀=T₀, Z=Z, kalg=kalg, falg=falg, natol=natol, nrtol=nrtol, nalg=nalg, bzkind=bzkind,
                μlims=μlims, ν=ν, nsp=nsp, Ωlims=Ωlims, atol=atol, rtol=rtol, io=io, verb=verb)

    condinter = interpolateconductivitykw(vcomp=Inter(), t=t, t′=t′, Δ=Δ, T=T, T₀=T₀, Z=Z, kalg=kalg, falg=falg, natol=natol, nrtol=nrtol, nalg=nalg, bzkind=bzkind,
                μlims=μlims, ν=ν, nsp=nsp, Ωlims=Ωlims, atol=atol, rtol=rtol, io=io, verb=verb)

    lines!(ax, ustrip.(Ωs), ustrip.(scalarize.(cond.(Ωs))), color=:black, label="whole")
    lines!(ax, ustrip.(Ωs), ustrip.(scalarize.(condintra.(Ωs))), color=cintra, label="inter-band")
    lines!(ax, ustrip.(Ωs), ustrip.(scalarize.(condinter.(Ωs))), color=cinter, label="inter-band")
    axislegend(ax)

    kps = KPathSegment(kpi.basis, kpi.kpaths[1], kpi.labels[1], kpi.setting)
    kloc = cumdists([bz.B*u"Å" * k for k in kps.kpath])
    ωlims = sort((-4t - chempot, 4t - chempot))
    freqs = range(ustrip.(ωlims)..., length=1000)

    structurelayout = GridLayout(nrow=4, ncol=2, parent=fig)

    axintra = Axis(fig,
    xlabel="Spectral density of $(scalarize_text) ($(unit(atol/t/det(bz.B))))", ylabel="ω (eV)",
    xticks=(kloc[collect(keys(kps.label))], map(string, values(kps.label))),
    limits = (extrema(kloc), extrema(freqs))
    )
    structurelayout[2,1] = axintra


    axinter = Axis(fig,
    xlabel="Spectral density of $(scalarize_text) ($(unit(atol/t/det(bz.B))))", ylabel="ω (eV)",
    xticks=(kloc[collect(keys(kps.label))], map(string, values(kps.label))),
    limits = (extrema(kloc), extrema(freqs))
    )
    structurelayout[4,1] = axinter


    kpathinterpplot!(axintra, kps, freqs, ocintra, alpha=1.0, densitycolormap=cintragrad) #densitycolormap=Makie.Reverse(:RdBu))
    kpathinterpplot!(axintra, kps, H)

    kpathinterpplot!(axinter, kps, freqs, ocinter, alpha=0.5, densitycolormap=cintergrad) #densitycolormap=Makie.Reverse(:RdBu))
    kpathinterpplot!(axinter, kps, H)


    insetintra = inset_axis!(fig, axintra; extent = (0.15, 0.45, 0.45, 0.75),
        limits=((1.0,2.5), (-0.05, 0.05)), xticklabelsvisible=false, xticksvisible=false, xgridvisible=false, ygridvisible=false)

    kpathinterpplot!(insetintra, kps, freqs, ocintra, alpha=1.0, densitycolormap=cintragrad) #densitycolormap=Makie.Reverse(:RdBu))
    kpathinterpplot!(insetintra, kps, H)

    insetinter = inset_axis!(fig, axinter; extent = (0.15, 0.45, 0.45, 0.75),
        limits=((1.0,2.0), (-0.2, 0.05)), xticklabelsvisible=false, xticksvisible=false, xgridvisible=false, ygridvisible=false)

    kpathinterpplot!(insetinter, kps, freqs, ocinter, alpha=1.0, densitycolormap=cintergrad) #densitycolormap=Makie.Reverse(:RdBu))
    kpathinterpplot!(insetinter, kps, H)

    axkintra = Axis(fig, ylabel="∫ $(scalarize_text)(k,ω) dω", limits=(extrema(kloc), nothing), ygridvisible=false)
    structurelayout[1,1] = axkintra
    linkxaxes!(axkintra, axintra)
    hidexdecorations!(axkintra, ticks = false, grid = false)


    axkinter = Axis(fig, ylabel="∫ $(scalarize_text)(k,ω) dω", limits=(extrema(kloc), nothing), ygridvisible=false)
    structurelayout[3,1] = axkinter
    linkxaxes!(axkinter, axinter)
    hidexdecorations!(axkinter, ticks = false, grid = false)

    # problem with nearly-degenerate eigenvectors is that
    dat_k_intra = [oc_kintegrand_intra(k, AutoBZCore.NullParameters()) for k in kpi]
    dat_k_inter = [oc_kintegrand_inter(k, AutoBZCore.NullParameters()) for k in kpi]


    lines!(axkintra, kloc, ustrip.(scalarize.(dat_k_intra)), color=cintra)
    lines!(axkinter, kloc, ustrip.(scalarize.(dat_k_inter)), color=cinter)

    axfintra = Axis(fig, xlabel="∫ $(scalarize_text)(k,ω) dk", xgridvisible=false)# limits=(nothing, extrema(freqs)))
    structurelayout[2,2] = axfintra
    linkyaxes!(axfintra, axintra)
    hideydecorations!(axfintra, ticks = false, grid = false)

    axfinter = Axis(fig, xlabel="∫ $(scalarize_text)(k,ω) dk", xgridvisible=false)# limits=(nothing, extrema(freqs)))
    structurelayout[4,2] = axfinter
    linkyaxes!(axfinter, axinter)
    hideydecorations!(axfinter, ticks = false, grid = false)
    
    oc_fintegrand_intra = interpolateconductivityk(vcomp=Intra(), μoffset=μintra, t=t, t′=t′, Δ=Δ, T=T, T₀=T₀, Z=Z, kalg=kalg, falg=falg, natol=natol, nrtol=nrtol, nalg=nalg, bzkind=bzkind,
            μlims=μlims, ν=ν, nsp=nsp, ωlims=ωlims, Ω=Ωintra, atol=atol, rtol=rtol, io=io, verb=verb)
    oc_fintegrand_inter = interpolateconductivityk(vcomp=Inter(), μoffset=μinter, t=t, t′=t′, Δ=Δ, T=T, T₀=T₀, Z=Z, kalg=kalg, falg=falg, natol=natol, nrtol=nrtol, nalg=nalg, bzkind=bzkind,
            μlims=μlims, ν=ν, nsp=nsp, ωlims=ωlims, Ω=Ωinter, atol=atol, rtol=rtol, io=io, verb=verb)

    # dat_f_intra = [oc_fintegrand_intra(f*unit(t), AutoBZCore.NullParameters()) for f in freqs]
    # dat_f_inter = [oc_fintegrand_inter(f*unit(t), AutoBZCore.NullParameters()) for f in freqs]

    lines!(axfintra, ustrip.(uconvert.(u"eV*Å^-1", scalarize.(oc_fintegrand_intra.(unit(t) .* freqs)))), freqs, color=cintra)
    lines!(axfinter, ustrip.(uconvert.(u"eV*Å^-1", scalarize.(oc_fintegrand_inter.(unit(t) .* freqs)))), freqs, color=cinter)


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
