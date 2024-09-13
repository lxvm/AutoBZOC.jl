using AutoBZOC, Unitful, LinearAlgebra

mkpath("kpathdensity")
make(;
    target=:kpathdensity,
    cache_dir="kpathdensity",
    figure_path = "figs_t2g",
    scalar_func = x -> tr(real(AutoBZOC.getval(x))),
    T=256.0u"K",
    # T=100.0u"K",
    quad_σ_k = AutoBZOC.IAI(AutoBZOC.AuxQuadGKJL(order=4)),
    quad_σ_ω = AutoBZOC.AuxQuadGKJL(order=4),
    interp_Ω = true,
    interp_ω = true,
    atol_σ = 1e-2u"Å^-1",
    rtol_σ = 1e-3,
    N_k = 4000,
    ylims = (0, 50),
    chempot = AutoBZOC.findchempot,
    config_vcomp = (
        (vcomp=AutoBZOC.Whole(), label="total",     color=:black,   densitycolormap=nothing,Ω=0.0u"eV", plot_trace=true,    plot_density=false, plot_ibz=false),
        (vcomp=AutoBZOC.Intra(), label="intraband", color=:orange,  densitycolormap=AutoBZOC.alphacolor.(reverse(AutoBZOC.colormap("RdBu", 256; mid=0.5, c=0.8, s=0.8, dcolor1=0.5)), vcat(sqrt.(range(0.5, 1.0, length=128)), ones(128))),  Ω=0.0u"eV", plot_trace=true,    plot_density=true,  plot_ibz=false),
        (vcomp=AutoBZOC.Inter(), label="interband", color=:green,   densitycolormap=AutoBZOC.alphacolor.(reverse(AutoBZOC.colormap("RdBu", 256; mid=0.5, c=0.8, s=0.8, dcolor1=0.5)), vcat(sqrt.(range(0.5, 1.0, length=128)), ones(128))),  Ω=0.4u"eV", plot_trace=true,    plot_density=true,  plot_ibz=true),
    ),
    series_kws = (; linewidth=8, color=[:black, :black, :black]),
    theme = merge(AutoBZOC.theme_latexfonts(), AutoBZOC.default.theme),
)
