module AutoBZOC

using LinearAlgebra
using NonlinearSolve
using AutoBZCore
using Unitful, UnitfulAtomic

export model, fig3a

const default = (;
    scalarize = real∘tr,
    scalarize_text = "Tr σ",
    t = -0.25u"eV",
    t′ = 0.05u"eV",
    Δ = 0.0u"eV",
    nalg = Falsi(),
    natol = 1e-5,
    nrtol = 1e-5,
    μlims = (-2.0u"eV", 2.0u"eV"),
    Ωintra = 0.0u"eV",
    Ωinter = 0.4u"eV",
    Tseries = ntuple(n -> 2.0 ^ (9-n) * u"K", 1),
    T = 100.0u"K",
    T₀ = 300.0u"K",
    Z = 0.5,
    Ωlims = (0.0u"eV", 1.0u"eV"),
    σatol = 0.0u"eV^2*Å^-1",
    σrtol = 1e-4,
    tolratio=100,
    ν = 1.0,
    nsp = 2,
    falg = AuxQuadGKJL(),
    kalg = PTR(npt=100),
    bzkind = CubicSymIBZ(),
    cintra = :orange,
    cinter = :green,
    prec = Float64,
    Nk = 1000,
    Nω = 1000,
    NΩ = 2000,
)
# Float32 precision not yet working

include("makieplots.jl")
include("model.jl")
include("density.jl")
include("conductivity.jl")
include("fig3.jl")
include("fig3a.jl")

end # module AutoBZOC
