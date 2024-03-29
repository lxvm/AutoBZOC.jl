struct SymIAI{T<:IAI} <: AutoBZCore.AutoBZAlgorithm
    alg::T
end
SymIAI(algs...) = SymIAI(IAI(algs...))

function AutoBZCore.bz_to_standard(bz::SymmetricBZ, alg::SymIAI)
    return AutoBZCore.bz_to_standard(bz, alg.alg)
end

function AutoBZCore.do_solve_autobz(bz_to_standard, f, bz, p, bzalg::SymIAI, cacheval; _kws...)
    bz_, dom, alg = bz_to_standard(bz, bzalg)
    j = abs(det(bz_.B))  # rescale tolerance to (I)BZ coordinate and get the right number of digits
    kws = NamedTuple(_kws)
    kws_ = haskey(kws, :abstol) ? merge(kws, (abstol=kws.abstol / j,)) : kws

    g = if f isa FourierIntegrand
        f.nest === nothing || error("not implemented")
        __f = (args...; kwargs...) -> begin
            out = f.f.f(args...; kwargs...)
            return AutoBZCore.symmetrize(f, bz, out)
        end
        _f = ParameterIntegrand{typeof(__f)}(__f, f.f.p)
        FourierIntegrand(_f, f.w)
    else
        error("not implemented")
    end
    sol = AutoBZCore.do_solve(g, dom, p, alg, cacheval; kws_...)
    # TODO find a way to throw a warning when constructing the problem instead of after a solve
    SymRep(f) isa UnknownRep && !(bz_ isa FullBZ) && !(sol.u isa TrivialRepType) && begin
        @warn AutoBZCore.WARN_UNKNOWN_SYMMETRY
        fbz = AutoBZCore.SymmetricBZ(bz_.A, bz_.B, lattice_bz_limits(bz_.B), nothing)
        _cacheval = AutoBZCore.init_cacheval(f, fbz, p, bzalg)
        return AutoBZCore.do_solve(f, fbz, p, bzalg, _cacheval; _kws...)
    end
    return AutoBZCore.IntegralSolution(sol.u * j, sol.resid * j, sol.retcode, sol.numevals)
end
