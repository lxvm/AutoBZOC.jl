using AutoBZ

# h, Σ, and μ are fixed parameters
# ω₁ and ω₂ are interpolation parameters
function transport_solver(; μ, kws...)
    (; model, selfenergy, quad_Γ_k, atol_Γ, rtol_Γ, gauge, vcomp, coord, nworkers, auxfun) = merge(default, NamedTuple(kws))

    h, bz, info_model = model(; kws..., gauge=Wannier())
    Σ, info_selfenergy = selfenergy(; kws...)
    info = (; model=info_model, selfenergy=info_selfenergy, μ, gauge, vcomp, coord, quad_Γ_k, atol_Γ, rtol_Γ, auxfun)

    hv = GradientVelocityInterp(h, bz.A; gauge, vcomp, coord)
    w = AutoBZCore.workspace_allocate_vec(hv, AutoBZCore.period(hv), Tuple(nworkers isa Int ? fill(nworkers, ndims(hv)) : nworkers))
    integrand = if auxfun === nothing
        TransportDistributionIntegrand(w; Σ, μ)
    else
        AuxTransportDistributionIntegrand(w, auxfun; Σ, μ)
    end
    Γ = IntegralSolver(integrand, bz, quad_Γ_k; abstol=atol_Γ, reltol=rtol_Γ)
    return Γ, info
end
