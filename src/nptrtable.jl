function nptrtable(; config_quad_breakeven, kws...)
    @show [   (begin
        μ, = findchempot(; kws..., T)
        stats, = benchmark_conductivity(; kws..., config_bench..., atol_σ, T, μ)
        stats.min.value.npt
        end)
        for (; config_bench, series_T, series_atol_σ, plot_kws) in config_quad_breakeven for (T, atol_σ) in zip(series_T, series_atol_σ) if config_bench.quad_σ_k isa AutoPTR]
end
