using GLMakie
using Brillouin

function fig_ibz(; colormap=:Spectral, Nk=30, sgnum=74, kws...)

    (; Δseries, Ωlims, σudisplay, σufactor, nsp, ndim) = merge(default, NamedTuple(kws))
    h, bz = t2gmodel(; kws..., gauge=Wannier())

    μ = findchempot(; kws...)
    shift!(h, μ)
    oc_integrand = OpticalConductivityIntegrand(AutoBZ.lb(Σ), AutoBZ.ub(Σ), σfalg, hv, Σ, β; abstol=atol/nsyms(bz), reltol=rtol)

    Δmax = maximum(Δseries)
    Δzero = zero(eltype(Δseries))
    Δmin = minimum(Δseries)

    kp = irrfbz_path(sgnum, eachcol(bz.A))

    fig = Figure(resolution=(800,600))
    ax1 = Axis3(fig[1,1])
    hidedecorations!(ax1)
    hidespines!(ax1)
    plot!(ax1, kp)

    ax2 = Axis3(fig[1,2])
    hidedecorations!(ax2)
    hidespines!(ax2)
    plot!(ax2, kp)
    polx = σ -> real(σ[1,1])
    poly = σ -> real(σ[2,2])
    H_k_Δ = map((Δmin, Δzero, Δmax)) do Δ
        evalHk(; kws..., Δ)
    end
    data = map(H_k_Δ) do H_k
        [oc_integrand(v, AutoBZCore.NullParameters()) for v in H_k]
    end
    datax = map(data) do σ
        polx.(σ)
    end
    datay = map(data) do σ
        poly.(σ)
    end

    volume!(ax1, datax; algorithm=:mip, colormap)
    volume!(ax2, datay; algorithm=:mip, colormap)
end