using GLMakie
using Brillouin
using SymmetryReduceBZ
using Polyhedra: Polyhedron, Mesh, points, polyhedron, vrep
using FourierSeriesEvaluators: workspace_evaluate!, workspace_contract!

function fig_ibz(; kpath=sgnum_path, colormap=:Spectral, N_k=30, sgnum=76, whichsym=13, pol=1,
    ibzcolor = :green, center = false, center_half = 0, config_ibz,
    alpharamp = range(0,1,length=256), cache_file_values_cond_k_cfs="cache-values-cond-k-cfs.jld2", kws...)
    (; T, chempot, scalar_func, vcomp, ndim, prec, cache_dir, nthreads) = merge(default, NamedTuple(kws))

    @assert ndim == 3

    β = invtemp(; kws..., T)

    kp = kpath(; sgnum, kws...)
    ibz = polyhedron(vrep(unique(values(kp.points))))
    klat = Iterators.product(ntuple(n -> AutoSymPTR.ptrpoints(prec, N_k), ndim)...)
    Kmask = fill(true, size(klat)) # in.(collect.(klat), Ref(ibz))
    limits = map(map(i -> extrema(p[i] for p in points(ibz)), ntuple(identity,ndim))) do (a, b)
        len = b - a
        (a-len/10, b+len/10)
    end
    limits = center ? ntuple(_ -> (-0.3, 0.5/1.1), 3) : ntuple(_ -> (0.0, 1.0), 3)

    fig = Figure(resolution=(3200,1600), figure_padding=0.0)
    mymap = Colors.alphacolor.(Makie.to_colormap(cgrad([ibzcolor, ibzcolor], 100)), alpharamp)
    mymap[1] = RGBAf(0,0,0,0)
    i = 0
    for (; Ω, vcomp, label, plot_kws) in config_ibz
        i += 1
        μ, = chempot(; T, kws...)
        σ_k, info_k = conductivity_solver(; kws..., T, μ, vcomp,
                                    bandwidth_bound=Ω, choose_kω_order=(alg...)->true,
                                    lims_Σ=fermi_window_limits(Ω, β))
        w = σ_k.f.w
        hv_k = AutoBZCore.FourierPTR(w, prec, Val(ndims(w.series)), N_k).s
        p_k = [AutoBZCore.MixedParameters(; hv_k=FourierValue(k, hv), Ω=prec(Ω),
                    getfield(σ_k.f.f.p, :kwargs)...) for (hv,k) in zip(hv_k,klat)]
        id_k = string((; info_k..., quad_σ_k=nothing, p=hash(p_k)))

        cache_path = joinpath(cache_dir, cache_file_values_cond_k_cfs)
        # there is no api to access the inner integral so this is inherently unstable
        # would need a nested integrand type to accomplish that
        dat = cache_batchsolve(σ_k.f.f.f.solver, p_k, cache_path, id_k, nthreads)
        σ_max = maximum(σ -> maximum(scalar_func(σ)), dat)
        σ_min = minimum(σ -> minimum(scalar_func(σ)), dat)
        @show σ_max σ_min
        data_σ_k = [ (y = scalar_func(σ)/σ_max; keep ? y : zero(y)) for (keep,σ) in zip(Kmask,dat) ]

        ax = Axis3(fig[1,i]; azimuth=2pi/3, elevation=-pi/3, limits)
        hidedecorations!(ax)
        hidespines!(ax)
        plot!(ax, kp; textkws = (; fontsize=96, strokewidth = 5), linewidth = 8, markersize = 50)
        # mesh!(ax, Mesh(ibz), alpha=0.2)
        volume!(ax, (center ? ntuple(n -> n == center_half ? klat.iterators[n][(end-div(size(data_σ_k,n),2)+1):end] .- 0.5 : klat.iterators[n] .- 0.5, 3) : klat.iterators)..., center ? circshift(data_σ_k, map(l -> div(l,2), size(data_σ_k)))[ntuple(n -> n == center_half ? (lastindex(data_σ_k,n)-div(size(data_σ_k,n),2)+1:(lastindex(data_σ_k,n))) : (firstindex(data_σ_k,n):lastindex(data_σ_k,n)), 3)...] : data_σ_k; plot_kws...)
        Label(fig[2,i], label; fontsize=96)

    end
    for j in 1:i
        colsize!(fig.layout, j, Relative(1/i))
    end
    rowsize!(fig.layout, 1, Relative(6/7))
    rowsize!(fig.layout, 2, Relative(1/7))


    return fig
end


function fig_ibz2(;colormap=:Spectral, N_k=30, sgnum=76, whichsym=13, pol=1,
    ibzcolor = :green, center = false, center_half = 0, config_ibz,
    alpharamp = range(0,1,length=256), cache_file_values_cond_k_cfs="cache-values-cond-k-cfs.jld2", kws...)
    (; T, model, chempot, scalar_func, vcomp, ndim, prec, cache_dir, nthreads) = merge(default, NamedTuple(kws))

    # @assert ndim == 2

    β = invtemp(; kws..., T)

    klat = Iterators.product(ntuple(n -> AutoSymPTR.ptrpoints(prec, N_k), 2)...)
    Kmask = fill(true, size(klat)) # in.(collect.(klat), Ref(ibz))

    fig = Figure(resolution=(3200,1600), figure_padding=0.0)
    mymap = Colors.alphacolor.(Makie.to_colormap(cgrad([ibzcolor, ibzcolor], 100)), alpharamp)
    mymap[1] = RGBAf(0,0,0,0)
    i = 0
    for (; Ω, vcomp, label, plot_kws) in config_ibz
        i += 1
        μ, = chempot(; T, kws...)
        σ_k, info_k = conductivity_solver(; kws..., T, μ, vcomp,
                                    bandwidth_bound=Ω, choose_kω_order=(alg...)->true,
                                    lims_Σ=fermi_window_limits(Ω, β))
        w = σ_k.f.w
        hv_k = AutoBZCore.FourierPTR(w, prec, Val(ndims(w.series)), N_k).s
        p_k = [AutoBZCore.MixedParameters(; hv_k=FourierValue(k, hv), Ω=prec(Ω),
                    getfield(σ_k.f.f.p, :kwargs)...) for (hv,k) in zip(hv_k,klat)]
        id_k = string((; info_k..., quad_σ_k=nothing, p=hash(p_k)))

        cache_path = joinpath(cache_dir, cache_file_values_cond_k_cfs)
        # there is no api to access the inner integral so this is inherently unstable
        # would need a nested integrand type to accomplish that
        dat = cache_batchsolve(σ_k.f.f.f.solver, p_k, cache_path, id_k, nthreads)
        σ_max = maximum(σ -> maximum(scalar_func(σ)), dat)
        σ_min = minimum(σ -> minimum(scalar_func(σ)), dat)
        @show σ_max σ_min
        data_σ_k = [ (y = scalar_func(σ)/σ_max; keep ? y : zero(y)) for (keep,σ) in zip(Kmask,dat) ]

        ax = Axis(fig[1,i]; aspect=1)
        # hidedecorations!(ax)
        # hidespines!(ax)
        # plot!(ax, kp; textkws = (; fontsize=96, strokewidth = 5), linewidth = 8, markersize = 50)
        # mesh!(ax, Mesh(ibz), alpha=0.2)
        heatmap!(ax, (center ? ntuple(n -> n == center_half ? klat.iterators[n][(end-div(size(data_σ_k,n),2)+1):end] .- 0.5 : klat.iterators[n] .- 0.5, 2) : klat.iterators)..., center ? circshift(data_σ_k, map(l -> div(l,2), size(data_σ_k)))[ntuple(n -> n == center_half ? (lastindex(data_σ_k,n)-div(size(data_σ_k,n),2)+1:(lastindex(data_σ_k,n))) : (firstindex(data_σ_k,n):lastindex(data_σ_k,n)), 2)...] : data_σ_k; plot_kws...)
        Label(fig[2,i], label; fontsize=96)

        for i in 1:6
            _,bz, = model(; kws..., whichperm=i, bzkind=IBZ(3), kz=(i==2 || i==5) ? -0.2 : 0.2)
            # @show bz.lims
            pts = bz.lims.vert
            # @show size(pts)
            lines!(ax, (push!(pts[:,1], pts[1,1])), (push!(pts[:,2], pts[1,2])); linewidth=8)
        end

    end
    for j in 1:i
        colsize!(fig.layout, j, Relative(1/i))
    end
    rowsize!(fig.layout, 1, Relative(6/7))
    rowsize!(fig.layout, 2, Relative(1/7))


    return fig
end

using GLMakie
using Brillouin
using SymmetryReduceBZ
using Polyhedra: Polyhedron, Mesh, points, polyhedron, vrep
using FourierSeriesEvaluators: workspace_evaluate!, workspace_contract!

function fig_ibz_resid(; η, kpath=sgnum_path, colormap=:Spectral, N_k=30, sgnum=76, whichsym=13, pol=1,
    ibzcolor = :green, center = false, center_half = 0, config_ibz,
    alpharamp = range(0,1,length=256), cache_file_values_cond_k_cfs="cache-values-cond-k-cfs.jld2", kws...)
    (; T, chempot, scalar_func, vcomp, ndim, prec, cache_dir, nthreads) = merge(default, NamedTuple(kws))

    @assert ndim == 3

    β = invtemp(; kws..., T)

    kp = kpath(; sgnum, kws...)
    ibz = polyhedron(vrep(unique(values(kp.points))))
    klat = Iterators.product(ntuple(n -> AutoSymPTR.ptrpoints(prec, N_k), ndim)...)
    Kmask = fill(true, size(klat)) # in.(collect.(klat), Ref(ibz))
    limits = map(map(i -> extrema(p[i] for p in points(ibz)), ntuple(identity,ndim))) do (a, b)
        len = b - a
        (a-len/10, b+len/10)
    end
    limits = center ? ntuple(_ -> (-0.3, 0.5/1.1), 3) : ntuple(_ -> (0.0, 1.0), 3)

    fig = Figure(resolution=(3200,1600), figure_padding=0.0)
    mymap = Colors.alphacolor.(Makie.to_colormap(cgrad([ibzcolor, ibzcolor], 100)), alpharamp)
    mymap[1] = RGBAf(0,0,0,0)
    i = 0
    for (; Ω, vcomp, label, plot_kws) in config_ibz
        i += 1
        μ, = chempot(; T, kws...)
        σ_k, info_k = conductivity_solver(; kws..., T, μ, vcomp, gauge=Hamiltonian(),
                                    bandwidth_bound=Ω, choose_kω_order=(alg...)->true,
                                    lims_Σ=fermi_window_limits(Ω, β))
        w = σ_k.f.w
        hv_k = AutoBZCore.FourierPTR(w, prec, Val(ndims(w.series)), N_k).s

        # there is no api to access the inner integral so this is inherently unstable
        # would need a nested integrand type to accomplish that
        # dat = cache_batchsolve(σ_k.f.f.f.solver, p_k, cache_path, id_k, nthreads)
        dat = [approx_ocfreqintegral(η, hv, Ω, μ, β) for hv in hv_k]
        σ_max = maximum(σ -> maximum(scalar_func(σ)), dat)
        data_σ_k = [ (y = scalar_func(σ)/σ_max; keep ? y : zero(y)) for (keep,σ) in zip(Kmask,dat) ]

        ax = Axis3(fig[1,i]; azimuth=2pi/3, elevation=-pi/3, limits)
        hidedecorations!(ax)
        hidespines!(ax)
        plot!(ax, kp; textkws = (; fontsize=96, strokewidth = 5), linewidth = 8, markersize = 50)
        # mesh!(ax, Mesh(ibz), alpha=0.2)
        volume!(ax, (center ? ntuple(n -> n == center_half ? klat.iterators[n][(end-div(size(data_σ_k,n),2)+1):end] .- 0.5 : klat.iterators[n] .- 0.5, 3) : klat.iterators)..., center ? circshift(data_σ_k, map(l -> div(l,2), size(data_σ_k)))[ntuple(n -> n == center_half ? (lastindex(data_σ_k,n)-div(size(data_σ_k,n),2)+1:(lastindex(data_σ_k,n))) : (firstindex(data_σ_k,n):lastindex(data_σ_k,n)), 3)...] : data_σ_k; plot_kws...)
        Label(fig[2,i], label; fontsize=96)

    end
    for j in 1:i
        colsize!(fig.layout, j, Relative(1/i))
    end
    rowsize!(fig.layout, 1, Relative(6/7))
    rowsize!(fig.layout, 2, Relative(1/7))


    return fig
end

function _fermi_window(β, ω, Ω)
    return (1/(exp(β*ω)+1) - 1/(exp(β*(ω+Ω))+1))/Ω
end
function ocfreqintegral_2poles(η, ϵn, ϵm, Ω, μ, β)
    pi*η*(_fermi_window(β, ϵn+im*η-μ, Ω)/((ϵn+Ω-ϵm + im*η)^2 + η^2) + _fermi_window(β, ϵm-Ω+im*η-μ, Ω)/((ϵm-Ω-ϵn + im*η)^2 + η^2))
end
function approx_ocfreqintegral(η, (h, v), Ω, μ, β)
    residues = map(Iterators.product(h.values, h.values)) do (ϵn, ϵm)
        ocfreqintegral_2poles(η, ϵn, ϵm, Ω, μ, β)
    end
    return map(Iterators.product(v, v)) do (va, vb)
        sum(va .* vb .* residues)
    end
end
