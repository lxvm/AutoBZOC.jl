using AutoBZOC, Unitful

mkpath("oc_fermiliquid")
make(;
    target = :oc_fermiliquid,
    cache_dir="oc_fermiliquid",
    figure_path="figs_t2g",
    series_T=u"K" * [256.0, 181.0], #128.0, 90.5, 64.0, 45.3, 32.0, 22.6, 16.0],
    chempot = AutoBZOC.findchempot,
    lims_y = (0, 82),
    theme = merge(AutoBZOC.theme_latexfonts(), AutoBZOC.default.theme),
    inset_lims_x = (10, 400),
    inset_lims_y = (20, 4e4),
    atol_σ = 1e-2u"Å^-1",
    rtol_σ = 1e-4,
    # threads = 4,
    # nworkers = (1, 1, 30),
)