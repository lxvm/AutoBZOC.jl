module AutoBZOC

using LinearAlgebra
using NonlinearSolve
using AutoBZ
using Unitful, UnitfulAtomic

export make

include("model.jl")

"""
    default

A NamedTuple containing all the default parameters used for calculations.
Any of these can be overridden for a given function call (and its callees).

Some functions (e.g. for figures) also have separate arguments (e.g. colors),
and some functions also accept global parameters `io`, `verb` and `cachepath`.
Multithreading can be adjusted with `nworkers` and `batchthreads`.

!!! note "Single-precision arithmetic `prec = Float32`"
    Using single precision arithmetic can yield a noticeable (~2x) speedup over double
    precision calculations, however `eps(one(Float32))=1.1920929f-7` means that caution must
    be taken with tolerances. The default tolerances seem to work well for double precision
    arithmetic, however in single precision they may not converge without increasing rtol or
    setting a finite atol.
"""
const default = (;
    scalarize = real∘tr,
    scalarize_text = "Tr σ",
    model = t2gmodel,
    ndim = 3,
    t = -0.25u"eV",
    t′ = 0.05u"eV",
    Δ = 0.0u"eV",
    self_energy = fermi_liquid_self_energy,
    nalg = Falsi(),
    nfalg = AuxQuadGKJL(order=7),
    nkalg = IAI(AuxQuadGKJL(order=7)),
    natol = 1e-5,
    nrtol = 1e-5,
    μlims = (-2.0u"eV", 2.0u"eV"),
    Ωintra = 0.0u"eV",
    Ωinter = 0.4u"eV",
    Tseries = u"K" * [256.0, 181.0, 128.0, 90.5, 64.0],
    Δseries = u"eV" * range(-1, 1, length=21),
    Ωseries = u"eV" * [0.0, 0.4],
    T = 100.0u"K",
    T₀ = 300.0u"K",
    Z = 0.5,
    Ωlims = (0.0u"eV", 1.0u"eV"),
    σudisplay = u"kS/cm",
    σufactor = π*u"e_au^2 * ħ_au / ħ_au^2", # don't forget nsp/(2pi)^ndim. note ħ^-2 come from velocities
    σatol = 0.0u"Å^-1",
    σrtol = 1e-4,
    auxfun = AutoBZ.default_transport_auxfun,
    σauxatol = 1e-3u"Å^-1",
    σauxrtol = 1e-3,
    interptolratio=100,
    ν = 1.0,
    nsp = 2,
    σfalg = AuxQuadGKJL(order=7),
    σkalg = IAI(AuxQuadGKJL(order=7)),
    bzkind = CubicSymIBZ(),
    prec = Float64,
    Nk = 1000,
    Nω = 1000,
    NΩ = 2000,
    gauge = Hamiltonian(),
    vcomp = Whole(),
    coord = Cartesian(),
    nworkers = 1,
    interp = false,
    nsample = 3,
    estkalg=PTR(npt=100),
    estfalg=QuadratureFunction(npt=100),
)

include("makieplots.jl")
include("density.jl")
include("conductivity.jl")
include("auxconductivity.jl")
include("fig3.jl")
include("fig3a.jl")
include("fig3_aux.jl")
include("fig3a_aux.jl")
include("fig_breakeven.jl")
include("fig_breakeven_aux.jl")
include("fig_cfs.jl")
include("fig_ibz.jl")
include("make.jl")

end # module AutoBZOC
