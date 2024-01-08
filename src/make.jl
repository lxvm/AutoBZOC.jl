using Logging

function do_make(; figure_path=joinpath(pwd(), "figs"), target=:all, io=stderr, min_level=Debug, kws...)
    with_logger(SimpleLogger(io, min_level)) do
        mkpath(figure_path)

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

        if target == :cfs || target == :all
            fc = fig_cfs(; kws...)
            save(joinpath(figure_path, "cfs.png"), fc)
        end

        if target == :ibz || target == :all
            fc = fig_ibz(; kws...)
            save(joinpath(figure_path, "ibz.png"), fc)
        end

    end
    return
end

# multithreading by default
make(; kws...) = do_make(;
    nthreads=Threads.nthreads(),
    kws...,
)

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
