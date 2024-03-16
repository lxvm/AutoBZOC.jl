module AutoBZOC

using LinearAlgebra
using SimpleNonlinearSolve
using AutoBZ
using Unitful, UnitfulAtomic
using GLMakie
using LaTeXStrings

export make

include("model.jl")

getval(x) = x
getval(x::AuxValue) = x.val

getaux(x) = x
getaux(x::AuxValue) = x.aux

_unit_Lstr(u) = latexstring(
    "\\left(\\,",
    replace(string(u),
        r"\h+" => s" \\:",
        r"(\w+)\^(-?\d+)" => s"\\mathrm{\1}^{\2}",
        r"(\w+)" => s"\\mathrm{\1}",
    ),
    "\\,\\right)")

"""
    default

A NamedTuple containing all the default parameters used for calculations.
Any of these can be overridden for a given function call (and its callees).

Some functions (e.g. for figures) also have separate arguments (e.g. colors).
Multithreading can be enabled with `nworkers` and `batchthreads`.

!!! note "Single-precision arithmetic `prec = Float32`"
    Using single precision arithmetic can yield a noticeable (~2x) speedup over double
    precision calculations, however `eps(one(Float32))=1.1920929f-7` means that caution must
    be taken with tolerances. The default tolerances seem to work well for double precision
    arithmetic, however in single precision they may not converge without increasing rtol or
    setting a finite atol.
"""
default = (;
    scalar_func = tr∘real∘getaux,
    scalar_text = L"\mathrm{Tr}~\sigma",
    bzkind = CubicSymIBZ(),
    kpath = cubic_path,
    model = t2g_model,
    t = -0.25u"eV",
    t′ = 0.05u"eV",
    Δ = 0.0u"eV",
    ndim = 3,
    gauge = Hamiltonian(),
    vcomp = Whole(),
    coord = Cartesian(),
    selfenergy = fermiliquid_selfenergy,
    lims_Σ = (-Inf*u"eV", Inf*u"eV"),
    T = 100.0u"K",
    T₀ = 300.0u"K",
    Z = 0.5,
    quad_g_k = IAI(AuxQuadGKJL(order=7)),
    atol_g = 1e-5u"Å^-3 * eV^-1",
    rtol_g = 1e-5,
    lims_ω = (-2.0u"eV", 2.0u"eV"),
    ν = 1.0,
    nsp = 2,
    root_n_μ = ITP(),
    quad_n_ω = AuxQuadGKJL(order=7),
    quad_n_k = IAI(AuxQuadGKJL(order=7)),
    choose_kω_order = default_kω_order,
    atol_n = 1e-5,
    rtol_n = 1e-5,
    lims_μ = (-2.0u"eV", 2.0u"eV"),
    quad_Γ_k = IAI(AuxQuadGKJL(order=7)),
    atol_Γ = 0e-0u"Å^(2-3)",
    rtol_Γ = 1e-5,
    quad_σ_ω = AuxQuadGKJL(order=7),
    quad_σ_k = IAI(AuxQuadGKJL(order=7)),
    auxfun = nothing,
    atol_σ = 0e-0u"Å^(2-3)",
    rtol_σ = 1e-4,
    lims_Ω = (0.0u"eV", 1.0u"eV"),
    unit_σ = u"kS/cm",
    factor_σ = π*u"e_au^2 * ħ_au / ħ_au^2", # don't forget nsp/(2pi)^ndim. note ħ^-2 come from velocities
    quadest_σ_k = PTR(npt=100),
    quadest_σ_ω = QuadratureFunction(npt=100),
    prec = Float64,
    interp_tolratio=100,
    interp_μ = false,
    interp_ω = true,
    interp_k = false,
    interp_Ω = true,
    N_k = 1000,
    N_ω = 1000,
    N_Ω = 2000,
    series_T = u"K" * [256.0, 181.0, 128.0, 90.5, 64.0],
    series_Δ = u"eV" * range(-1, 1, length=21),
    config_vcomp = (
        (vcomp=Whole(), label="whole",      color=:black,   densitycolormap=nothing,Ω=0.0u"eV", plot_trace=true,    plot_density=false, plot_ibz=false),
        (vcomp=Intra(), label="intra-band", color=:orange,  densitycolormap=Makie.Reverse(:RdBu),  Ω=0.0u"eV", plot_trace=true,    plot_density=true,  plot_ibz=false),
        (vcomp=Inter(), label="inter-band", color=:green,   densitycolormap=Makie.Reverse(:RdBu),  Ω=0.4u"eV", plot_trace=true,    plot_density=true,  plot_ibz=true),
    ),
    config_quad_breakeven = (;
        algs = (
            IAI(AuxQuadGKJL(order=4)),
            IAI(AuxQuadGKJL(order=5)),
            # IAI(AuxQuadGKJL(order=6)),
            # IAI(AuxQuadGKJL(order=7)),
            # IAI(AuxQuadGKJL(order=8)),
            AutoPTR(nmin=1, nmax=typemax(Int)),
        ),
        series = (
            # (; fun = a -> 1/a,      label = "1/η",      factor_t=1e-7, factor_numevals=1e1),
            # (; fun = a -> log(1/a), label = "log(1/η)", factor_t=1e-5, factor_numevals=1e2),
            # (; fun = a -> 1/a^2,      label = "1/η²",      factor_t=1e-7, factor_numevals=1e1),
            # (; fun = a -> log(1/a)^2, label = "log(1/η)²", factor_t=1e-4, factor_numevals=1e4),
            (; fun = a -> 1/a^3,      label = "1/η³",      factor_t=1e-7, factor_numevals=1e1),
            (; fun = a -> log(1/a)^3, label = "log(1/η)³", factor_t=1e-2, factor_numevals=1e5),
        )
    ),
    nsample = 3,
    nworkers = 1,
    nthreads = 1,
    cache_dir = ".",
    theme = merge(
        theme_latexfonts(),
        Theme(;
            Axis = (;
                xgridvisible=false,
                xtickalign = 1,
                ygridvisible=false,
                ytickalign = 1,
                yticklabelrotation=pi/2,
            ),
            fontsize = 16,
        ),
    ),
)

# the model of the material determines the Hamiltonian and self energy
# the caller has to supply the keyword arguments that we don't interpolate
# once these are set they should not be overwritten (AutoBZ allows them to be)
# so studies may interpolate/batch the remaining parameters but should create
# new problems in order to change the required keywords

# So there are three kinds of parameters
# 1. model parameters (e.g. h, Σ)
# 2. discrete tunable parameter (e.g. T)
# 3. interpolable parameters (e.g. μ, ω, Ω)

include("caching.jl")
include("trgloc.jl")
include("density.jl")
include("transport.jl")
include("conductivity.jl")
include("conductivity_test.jl")
include("conductivity_only.jl")
include("guess_dc_scaling.jl")

include("makieplots.jl")
include("fig_bands.jl")
include("fig3.jl")
include("fig3a.jl")
include("fig_breakeven.jl")
include("fig_cfs.jl")
include("fig_ibz.jl")
include("fig_dos.jl")
include("fig_err.jl")
include("make.jl")

end # module AutoBZOC
