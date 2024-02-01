using OffsetArrays, StaticArrays, FourierSeriesEvaluators
using Unitful, UnitfulAtomic
using Permutations
using LinearAlgebra
using AutoBZ
using Brillouin
using Random

function integerlattice_model(; kws...)
    (; t, ndim, gauge, prec, bzkind) = merge(default, NamedTuple(kws))
    C = OffsetArray(zeros(SMatrix{1,1,typeof(prec(t)),1},ntuple(_ -> 3, ndim)), repeat([-1:1], ndim)...)
    for i in 1:ndim, j in (-1, 1)
        C[CartesianIndex(ntuple(k -> k ≈ i ? j : 0, ndim))] = [t;;]
    end

    info = (; name="integerlattice", t, ndim, gauge, prec)
    d = ndim
    A = one(SMatrix{d,d,prec,d^2})
    bz = if bzkind isa IBZ
        @assert ndim == 3
        atom_species = ["Sr"]
        atom_pos = zeros(ndim)
        load_bz(bzkind, bz.A, bz.B, atom_species, atom_pos')
        # TODO change units of A, B
    else
        load_bz(bzkind, SMatrix(A) * u"Å")
    end
    return HamiltonianInterp(AutoBZ.Freq2RadSeries(FourierSeries(C, period=prec(real(2one(t)*pi)))); gauge), bz, info
end

function random_model(; seed, nband, nmode, kws...)
    (; t, ndim, gauge, prec, bzkind) = merge(default, NamedTuple(kws))
    ndim == 1 || error("multidimesional not implemented")
    Random.seed!(seed)
    d = ndim
    M = nmode
    T = SMatrix{nband,nband,typeof(prec(t)),nband^2}
    info = (; name="random", t, seed, nband, nmode, prec, bzkind)
    hm = Vector{T}(undef, 2M+1)
    # Hermitian means H_R = H_{-R}^\dagger
    for i in 0:M
        el = rand(T)*exp(-abs(i)) # exponentially decaying coefficients
        if i == 0
            hm[M+1] = el + el'
        else
            hm[M+1+i] = el
            hm[M+1-i] = el'
        end
    end
    bzkind isa FBZ || @warn "random Fourier series has no symmetries. For correctness use bzkind=FBZ()"
    A = one(SMatrix{d,d,prec,d^2})
    bz = load_bz(bzkind, SMatrix(A) * u"Å")
    return HamiltonianInterp(AutoBZ.Freq2RadSeries(FourierSeries(hm, period=prec(real(2one(t)*pi)), offset=-M-1)); gauge), bz, info
end


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

function t2g_model(; kws...)

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
    bz = if bzkind isa IBZ
        @assert ndim == 3
        atom_species = [
            "Sr",
            "V",
            "O",
            "O",
            "O",
        ]
        atom_pos = [
            0.0 0.0 0.0
            0.5 0.5 0.5
            0.0 0.5 0.5
            0.5 0.0 0.5
            0.5 0.5 0.0
        ]
        load_bz(bzkind, bz.A, bz.B, atom_species, atom_pos')
        # TODO change units of A, B
    else
        load_bz(bzkind, SMatrix(A) * u"Å")
    end
    return HamiltonianInterp(AutoBZ.Freq2RadSeries(FourierSeries(similar(H, SM) .= H; period=prec(real(2one(t)*pi)))); gauge), bz, info
end


function fermiliquid_selfenergy(; T, kws...)
    (; t, T₀, Z, prec, lims_Σ) = merge(default, NamedTuple(kws))
    η = map(T -> prec(uconvert(unit(t), T^2*u"k_au"*pi/(Z*T₀))), T)
    info = (; name=:fermiliquid, η, T, T₀, Z, prec, lims_Σ)
    return ConstScalarSelfEnergy(-im*η, map(prec, lims_Σ)...), info
end

function autobz_selfenergy(; file_selfenergy, offset_scattering, config_selfenergy=(;), kws...)
    (; prec, lims_Σ) = merge(default, NamedTuple(kws))
    Σ = load_self_energy(file_selfenergy; precision=prec, config_selfenergy...)
    lims_Σ = (prec(max(Σ.lb*u"eV", lims_Σ[1])), prec(min(Σ.ub*u"eV", lims_Σ[2])))
    info = (; name=:autobz, file=file_selfenergy, precision=prec, lims_Σ, offset_scattering, config_selfenergy...)
    return MatrixSelfEnergy(lims_Σ...) do ω
        Σ(prec(ω/u"eV"))*(one(prec)*u"eV") - (prec(offset_scattering)*im)*I
    end, info
end

# make T a required keyword so that the caller has to set it explicitly
function invtemp(; T, kws...)
    (; t, prec) = merge(default, NamedTuple(kws))
    return prec(1/uconvert(unit(t), u"k_au"*T))
end

function convergence(; kws...)
    (; model, selfenergy) = merge(default, NamedTuple(kws))
    Σ, = selfenergy(; kws...)
    η = AutoBZ.sigma_to_eta(Σ)
    h, = model(; kws...)
    vT = AutoBZ.velocity_bound(h)
    return AutoBZ.freq2rad(η/vT)
end

function wannier90_model(; seed, bzkind=FBZ(), kws...)
    (; gauge, prec) = merge(default, NamedTuple(kws))
    info = (; name=:wannier90, seed, gauge, bzkind, prec)
    h_, bz_ = load_wannier90_data(seed; gauge, bz=bzkind)
    f = AutoBZ.parentseries(h_)
    h = HamiltonianInterp(AutoBZ.Freq2RadSeries(FourierSeries(f.c * u"eV"; period=f.t, offset=f.o, deriv=f.a)); gauge)
    bz = SymmetricBZ(bz_.A * u"Å", bz_.B / u"Å", bz_.lims, bz_.syms)
    return h, bz, info
end

default_kω_order(alg_k, alg_ω) = !(alg_k isa AutoPTR || alg_k isa PTR)

function cubic_path(; kws...)
    (; model) = merge(default, NamedTuple(kws))

    _, bz, = model(; kws...)
    @assert LinearAlgebra.checksquare(bz.A) == 3
    pts = Dict{Symbol,SVector{3,Float64}}(
        :R => [0.5, 0.5, 0.5],
        :M => [0.5, 0.5, 0.0],
        :Γ => [0.0, 0.0, 0.0],
        :X => [0.0, 0.5, 0.0],
    )
    paths = [
        [:R, :Γ, :X, :M, :Γ],
    ]
    basis = Brillouin.KPaths.reciprocalbasis(collect(eachcol(bz.B'bz.A)))
    setting = Ref(Brillouin.LATTICE)
    return KPath(pts, paths, basis, setting)
end

function custom_path(; pts, paths, A, setting, kws...)
    basis = Brillouin.KPaths.reciprocalbasis(collect(eachcol(A)))
    return KPath(pts, paths, basis, Ref(setting))
end

function sgnum_path(; sgnum, kws...)
    (; model) = merge(default, NamedTuple(kws))

    _, bz, = model(; kws...)
    return irrfbz_path(sgnum, collect(eachcol(bz.B'bz.A)))
end

function cfs_path(; kws...)
    A = 2pi*I(3)
    irrfbz_path(sgnum, A)
    pts = Dict{Symbol,SVector{3,Float64}}(
        :Z => [0.0, 0.0, 0.5],
        :R => [0.5, 0.0, 0.5],
        :M => [0.5, 0.5, 0.0],
        :A => [0.5, 0.5, 0.5],
        :Γ => [0.0, 0.0, 0.0],
        :X => [0.5, 0.0, 0.0],
    )
    paths = [
        [:Γ, :X, :M, :Γ, :Z, :R, :A, :Z],
        [:X, :R],
        [:M, :A],
    ]
    basis = Brillouin.KPaths.reciprocalbasis(A)
    setting = Ref(Brillouin.LATTICE)
    return KPath(pts, paths, basis, setting)
end
