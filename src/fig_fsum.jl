function fig_fsum(; cache_file_interp_chempot="cache-interp-chempot.jld2", cache_file_interp_fsum="cache-interp-fsum.jld2", kws...)
    (; series_T, lims_μ, nsp, atol_μ, rtol_μ, atol_n, rtol_n, root_n_μ, N_n, unit_σ, factor_σ, ndim, quad_fsum_Ω, atol_fsum, rtol_fsum, scalar_func, cache_dir) = merge(default, NamedTuple(kws))
    fig = Figure()

    for T in series_T
        ρ, V, info_density = density_interp(; T, kws...)

        lims_n = (ρ(; μ=lims_μ[1]), ρ(; μ=lims_μ[2])) ./ V .* nsp
        cache_path_μ = joinpath(cache_dir, cache_file_interp_chempot)
        id_μ = string((; info_density..., atol_μ, rtol_μ, atol_n, rtol_n, root_n_μ))

        μ = cache_hchebinterp(lims_n..., atol_μ, rtol_μ, 1, cache_path_μ, id_μ) do n
            u = unit(eltype(lims_μ))
            map(n) do n
                prob = IntervalNonlinearProblem(lims_μ ./ u, (n, V, nsp)) do μ, (n, V, nsp)
                    upreferred(ρ(; μ=μ*u)/V*nsp - n)
                end
                u*rootsolve(prob, root_n_μ; abstol=atol_n, reltol=rtol_n)
            end
        end

        σ, info_σ = conductivity_solver(; μ=0.0u"eV", bandwidth_bound=1.0u"eV", kws...)
        cache_path_σ = joinpath(cache_dir, cache_file_interp_fsum)
        id_fsum = string((; info_σ, lims_μ, atol_fsum, rtol_fsum))
        fsum = cache_hchebinterp(lims_μ..., atol_fsum, rtol_fsum, 1, cache_path_σ, id_fsum) do μ
            map(μ) do μ
                prob = AutoBZCore.IntegralProblem((Ω, μ) -> σ(; Ω, μ), 0.0u"eV", Inf*u"eV", μ)
                AutoBZCore.solve(prob, quad_fsum_Ω).u
            end
        end
        ax = Axis(fig[1,1];)
        series_n = range(lims_n..., length=N_n)
        lines!(ax, series_n, n -> upreferred(scalar_func(fsum(μ(n)))*(nsp*factor_σ/(2pi)^ndim/unit_σ/u"eV")); label="$T")
    end
    axislegend()
    return fig
end