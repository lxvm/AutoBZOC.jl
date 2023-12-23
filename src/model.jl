using OffsetArrays, StaticArrays, FourierSeriesEvaluators
using Unitful, UnitfulAtomic
using Permutations
using LinearAlgebra
using AutoBZ


function ogmodel(; kws...)

    (; t, t′, Δ, ndim) = merge(default, NamedTuple(kws))
    d = ndim
    SM = SMatrix{d,d,typeof(t),d^2}
    A = zeros(SM, ntuple(_->3,d)...)
    H = OffsetArray(A, ntuple(_->-1:1,3)...)

    # intraband hoppings
    for i in 1:d
        idx = CartesianIndex((ntuple(j -> j==i ? 1 : 0, d)))
        h = StaticArrays.sacollect(SM, (m == n && m != i) ? t : zero(t) for m in 1:d, n in 1:d)
        H[ idx] = h
        H[-idx] = h'
    end

    # interband hoppings

    d == 3 || throw(ArgumentError("only d=3 implemented"))

    # related to all possible ways to connect orbital pairs along coordinate axes, accounting for
    # orientation with sign flips
    H[ 0, 1, 1] =  [ 0; 0; 0;; 0; 0;t′;; 0;t′; 0]
    H[ 0,-1,-1] =  [ 0; 0; 0;; 0; 0;t′;; 0;t′; 0]'
    H[ 0, 1,-1] = -[ 0; 0; 0;; 0; 0;t′;; 0;t′; 0]
    H[ 0,-1, 1] = -[ 0; 0; 0;; 0; 0;t′;; 0;t′; 0]'

    H[ 1, 0, 1] =  [ 0; 0;t′;; 0; 0; 0;;t′; 0; 0]
    H[-1, 0,-1] =  [ 0; 0;t′;; 0; 0; 0;;t′; 0; 0]'
    H[ 1, 0,-1] = -[ 0; 0;t′;; 0; 0; 0;;t′; 0; 0]
    H[-1, 0, 1] = -[ 0; 0;t′;; 0; 0; 0;;t′; 0; 0]'

    H[ 1, 1, 0] =  [ 0;t′; 0;;t′; 0; 0;; 0; 0; 0]
    H[-1,-1, 0] =  [ 0;t′; 0;;t′; 0; 0;; 0; 0; 0]'
    H[ 1,-1, 0] = -[ 0;t′; 0;;t′; 0; 0;; 0; 0; 0]
    H[-1, 1, 0] = -[ 0;t′; 0;;t′; 0; 0;; 0; 0; 0]'

    # crystal field splitting on 1st orbital
    H[CartesianIndex(ntuple(zero,d))] = StaticArrays.sacollect(SM, (m == n == 1) ? Δ : zero(Δ) for m in 1:d, n in 1:d)

    return FourierSeries(H, period=real(2one(t)*pi))
end

function t2gmodel(; kws...)

    (; t, t′, Δ, ndim, gauge, prec, bzkind) = merge(default, NamedTuple(kws))
    info = (; name=:t2g, ndim, t, t′, Δ, gauge, bzkind, prec)
    d = ndim
    SM = SHermitianCompact{d,typeof(prec(t)),StaticArrays.triangularnumber(d)}
    # SM = SMatrix{d,d,typeof(prec(t)),d^2}
    MM = MMatrix{d,d,typeof(prec(t)),d^2}
    A = MM[zero(MM) for _ in Iterators.product(ntuple(_->1:3,Val(d))...)]
    H = OffsetArray(A, ntuple(_->-1:1,Val(d))...)

    # intraband hoppings
    for i in 1:d
        idx = CartesianIndex((ntuple(j -> j==i ? 1 : 0, Val(d))))
        # h = StaticArrays.sacollect(SM, (m == n && m != i) ? t : zero(t) for m in 1:d, n in 1:d)
        for n in 1:d
            n == i && continue
            H[ idx][n,n] = t
            H[-idx][n,n] = conj(t)
        end
    end

    # interband hoppings
    # related to all possible ways to connect orbital pairs along coordinate axes, accounting for
    # orientation with sign flips
    for i in CartesianIndices(H)
        j = findfirst(iszero, i.I)
        j === nothing && continue
        count(==(1)∘abs, i.I) == d-1 || continue
        h = H[i]
        par = prod(i.I[n] for n in eachindex(i.I) if n != j; init=1)
        for k in PermGen(d-1)
            idx = CartesianIndex(ntuple(l -> k[l]>=j ? k[l]+1 : k[l], Val(d-1)))
            h[idx] = par*(sign(k)>0 ? t′ : conj(t′))
        end
    end

    # crystal field splitting on 1st orbital
    H[CartesianIndex(ntuple(zero,Val(d)))][1,1] += Δ
    # we want to deform the bz along the direction of CFS without changing det(bz.B)
    δ = exp(Δ/10t)
    A = one(MMatrix{d,d,prec,d^2}) * δ^(-1/(d-1))
    A[d,d] = δ

    # construct corresponding Brillouin zone
    # TODO: return InversionSymIBZ
    !iszero(Δ) && bzkind isa CubicSymIBZ && error("nonzero CFS breaks cubic symmetry in BZ, try bzkind=InversionSymIBZ()")
    bz = load_bz(bzkind, SMatrix(A) * u"Å")

    return HamiltonianInterp(AutoBZ.Freq2RadSeries(FourierSeries(similar(H, SM) .= H; period=prec(real(2one(t)*pi)))); gauge), bz, info
end

function fermi_liquid_scattering(; kws...)
    (; t, T, T₀, Z, prec) = merge(default, NamedTuple(kws))
    return map(T -> prec(uconvert(unit(t), T^2*u"k_au"*pi/(Z*T₀))), T)
end

function fermi_liquid_self_energy(; kws...)
    η = fermi_liquid_scattering(; kws...)
    return EtaSelfEnergy(η)
end

function inv_temp(; kws...)
    (; t, T, prec) = merge(default, NamedTuple(kws))
    return prec(1/uconvert(unit(t), u"k_au"*T))
end

function convergence(; kws...)
    (; model, self_energy) = merge(default, NamedTuple(kws))
    Σ = self_energy(; kws...)
    η = AutoBZ.sigma_to_eta(Σ)
    name, h, = model(; kws...)
    vT = AutoBZ.velocity_bound(h)
    return AutoBZ.freq2rad(η/vT)
end

function wannier90model(; seed, bzkind=FBZ(), kws...)
    (; gauge, prec) = merge(default, NamedTuple(kws))
    info = (; name=:wannier90, seed, gauge, bzkind, prec)
    h_, bz_ = load_wannier90_data(seed; gauge, bz=bzkind)
    f = AutoBZ.parentseries(h_)
    h = HamiltonianInterp(AutoBZ.Freq2RadSeries(FourierSeries(f.c * u"eV"; period=f.t, offset=f.o, deriv=f.a)); gauge)
    bz = SymmetricBZ(bz_.A * u"Å", bz_.B / u"Å", bz_.lims, bz_.syms)
    return h, bz, info
end
