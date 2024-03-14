function guess_dc_scaling(; series_T, kws...)
    σDC = map(series_T) do T
        μ, V, = findchempot(; kws..., T)
        σ, = conductivity_batchsolve(; kws..., μ, T, series_Ω=zero(μ))
        only(σ)
    end
    log_T = map(T -> log(ustrip(T)), series_T)
    X = hcat(ones(eltype(log_T), length(log_T)), log_T)
    log_σDC = map(σ -> log(ustrip(real(tr(σ)))), σDC)
    b, a = (X'X) \ X'log_σDC # linear fit to the log-log data
    return b, a, unit(eltype(eltype(σDC)))
end

function extrapolate_dc_scaling(; extrap_T, kws...)
    b, a, unit_σDC = guess_dc_scaling(; kws...)
    return exp.(a .* log.(ustrip.(extrap_T)) .+ b) .* unit_σDC
end