using AutoBZOC, Unitful

mkpath("bands")
make(;
    target=:bands,
    cache_dir="bands",
    figure_path = "figs_t2g",
    T=100.0u"K",
    chempot = AutoBZOC.chempot_manual,
    μ = 0.0u"eV",
)
make(;
    target=:bands,
    cache_dir="bands",
    figure_path = "figs_t2g_nohop",
    t′ = 0.0u"eV",
    T=100.0u"K",
    chempot = AutoBZOC.chempot_manual,
    μ = 0.0u"eV",
)

mkpath("dos_fermiliquid")
make(;
    target=:dos_fermiliquid,
    cache_dir="dos_fermiliquid",
    figure_path = "figs_t2g",
    series_T = u"K" * [500.0],
    # series_T = u"K" * [16.0],
    model = AutoBZOC.t2g_model,
    atol_g = 1e-3u"Å^-3 * eV^-1",
    rtol_g = 0.0,
    lims_ω = (-2.0u"eV", 2.0u"eV"),
    chempot = AutoBZOC.chempot_manual,
    μ = 0.0u"eV",
    gauge = AutoBZOC.Wannier(),
)
make(;
    target=:dos_fermiliquid,
    cache_dir="dos_fermiliquid",
    figure_path = "figs_t2g_nohop",
    t′ = 0.0u"eV",
    series_T = u"K" * [500.0],
    # series_T = u"K" * [16.0],
    model = AutoBZOC.t2g_model,
    atol_g = 1e-3u"Å^-3 * eV^-1",
    rtol_g = 0.0,
    lims_ω = (-2.0u"eV", 2.0u"eV"),
    chempot = AutoBZOC.chempot_manual,
    μ = 0.0u"eV",
    gauge = AutoBZOC.Wannier(),
)