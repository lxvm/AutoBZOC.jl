using Logging

function do_make(; figure_path=joinpath(pwd(), "figs"), target=:all, io=stderr, min_level=Debug, kws...)
    with_logger(SimpleLogger(io, min_level)) do
        mkpath(figure_path)

        if target == :bands || target == :all
            fb = fig_bands(; kws...)
            save(joinpath(figure_path, "bands.png"), fb)
        end

        if target == :dos || target == :all
            fg = fig_breakeven_trgloc(; kws...)
            save(joinpath(figure_path, "crossover_dos.png"), fg)
        end

        if target == :dos_fermiliquid || target == :all
            fd = fig_dos(; kws...)
            save(joinpath(figure_path, "dos_fermiliquid.png"), fd)
        end

        if target == :kpathdensity || target == :all
            f3a = fig3a(; kws...)
            save(joinpath(figure_path, "kpathdensity.png"), f3a)
        end

        if target == :oc_fermiliquid || target == :all
            f3 = fig3(; kws...)
            save(joinpath(figure_path, "oc_fermiliquid.png"), f3)
        end

        if target == :crossover || target == :all
            fb = fig_breakeven(; kws...)
            save(joinpath(figure_path, "crossover.png"), fb)
        end

        if target == :crossover_log || target == :all
            fl = fig_breakeven_log(; kws...)
            save(joinpath(figure_path, "crossover_log.png"), fl)
        end

        if target == :nptrtable || target == :all
            dat = nptrtable(; kws...)
            # writedlm(joinpath(figure_path, "nptrtable.txt"), dat)
        end

        if target == :cfs || target == :all
            fc = fig_cfs(; kws...)
            save(joinpath(figure_path, "cfs.png"), fc)
        end

        if target == :ibz || target == :all
            fi = fig_ibz(; kws...)
            save(joinpath(figure_path, "ibz.png"), fi)
        end

        if target == :auxerr || target == :all
            fe = fig_err(; kws...)
            save(joinpath(figure_path, "auxerr.png"), fe)
        end

        if target == :doserr || target == :all
            fed = fig_err_dos(; kws...)
            save(joinpath(figure_path, "doserr.png"), fed)
        end

        if target == :testerr || target == :all
            fet = fig_err_test(; kws...)
            save(joinpath(figure_path, "testerr.png"), fet)
        end

        if target == :onlyerr || target == :all
            feo = fig_err_only(; kws...)
            save(joinpath(figure_path, "onlyerr.png"), feo)
        end
    end
    return
end

# multithreading by default
function make(; kws...)
    (; theme) = merge(default, NamedTuple(kws))
    set_theme!(theme)
    do_make(;
        nthreads=Threads.nthreads(),
        kws...,
    )
end

# make for auxiliary functions
make_aux(; kws...) = make(;
    figure_path=joinpath(pwd(), "figs_aux"),
    auxfun = AutoBZ.default_transport_auxfun,
    atol_Γ = AuxValue(0e-0, 1e-3)*u"Å^-1",
    rtol_Γ = AuxValue(1e-5, 1e-3),
    atol_σ = AuxValue(0e-0, 1e-3)*u"Å^-1",
    rtol_σ = AuxValue(1e-4, 1e-3),
    kws...,
)

make_dos(; kws...) = make(;
    target = :dos,
    kws...,
)

# make with single-precision turned on by default
make32(; kws...) = make(;
    rtol_n = 1e-4,
    atol_Γ = 1e-5u"Å^-1",
    atol_σ = 1e-5u"Å^-1",
    kws...,
    prec = Float32,
)

make32_aux(; kws...) = make_aux(;
    rtol_n = 1e-4,
    atol_Γ = AuxValue(1e-5, 1e-3)*u"Å^-1",
    atol_σ = AuxValue(1e-5, 1e-3)*u"Å^-1",
    kws...,
    prec = Float32,
)

make32_dos(; kws...) = make_dos(;
    rtol_n = 1e-4,
    kws...,
    prec = Float32,
)
