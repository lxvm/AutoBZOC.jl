using AutoBZOC, Unitful, LinearAlgebra

mkpath("ibz")
make(;
    target=:ibz,
    cache_dir="ibz",
    figure_path = "figs_t2g",
    config_ibz = [
        (; vcomp = AutoBZOC.Intra(), Ω=0.0u"eV", label = AutoBZOC.L"(a) Intraband $\mathrm{Tr}~\sigma(\mathbf{k})$ at $\Omega$ = 0.0 eV", plot_kws=(; algorithm=:iso, isovalue=1.0, isorange=0.995, colormap=AutoBZOC.cgrad([:teal, :teal], 10))),
        (; vcomp = AutoBZOC.Inter(), Ω=0.4u"eV", label = AutoBZOC.L"(b) Interband $\mathrm{Tr}~\sigma(\mathbf{k})$ at $\Omega$ = 0.4 eV", plot_kws=(; algorithm=:iso, isovalue=1.0, isorange=0.995, colormap=AutoBZOC.cgrad([:teal, :teal], 10))),
    ],
    sgnum=221,
    N_k = 50,
    # N_k = 200,
    T=16.0u"K",
    chempot = AutoBZOC.findchempot,
    model=AutoBZOC.t2g_model,
    center = true,
    center_half = 3,
    ndim = 3,
    scalar_func = σ -> real(tr(σ)),
    theme = merge(AutoBZOC.theme_latexfonts(), AutoBZOC.Theme(; fontsize = 36)),
)