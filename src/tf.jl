# h and β are fixed parameters
# μ is an interpolation parameter
function tf_solver(; kws...)
    (; velocity, quad_tf_k, atol_tf, rtol_tf, nworkers) = merge(default, NamedTuple(kws))

    h, bz, info_model_velocity = velocity(; kws...)
    β = invtemp(; kws...)
    info = (; velocity=info_model_velocity, β, quad_tf_k, atol_tf, rtol_tf)

    w = AutoBZCore.workspace_allocate_vec(h, AutoBZCore.period(h), Tuple(nworkers isa Int ? fill(nworkers, ndims(h)) : nworkers))
    integrand = TransportFunctionIntegrand(w; β)
    # integrand = TrGlocIntegrand(w; Σ, μ)
    tf = IntegralSolver(integrand, bz, quad_tf_k; abstol=atol_tf, reltol=rtol_tf)
    return tf, info
end