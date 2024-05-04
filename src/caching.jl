
using JLD2
using HChebInterp
using AutoBZ
using SimpleNonlinearSolve

function cache_call(f, args, kws, cache_path, key; call="f", use_cache=true, _...)
    return jldopen(cache_path, "a+") do fn
        if !haskey(fn, key) || !use_cache
            @debug "$call call made"
            stats = @timed f(args...; kws...)
            use_cache ? (fn[key] = stats) : stats
        else
            @debug "$call call found"
            fn[key]
        end
    end
end

function cache_hchebinterp(batchf, a, b, atol, rtol, initdiv, cache_path, key; call="hchebinterp", max_batch=1, kws...)
    cnt::Int = 0
    f = BatchFunction(; max_batch) do x
        cnt += nbatch = length(x)
        dat = @timed batchf(x)
        @debug "$call batch" batch_elapsed=dat.time batch_samples=nbatch
        dat.value
    end
    stats = cache_call(hchebinterp, (f, a, b), (; atol, rtol, initdiv, maxevals=1000), cache_path, key; call, kws...)
    cnt > 0 && @debug "$call summary" elapsed=stats.time samples=cnt
    return stats.value
end

function cache_batchsolve(solver, params, cache_path, key, nthreads; call="batchsolve", kws...)
    called::Bool = false
    batchsolve = (args...; kws...) -> (called = true; AutoBZCore.batchsolve(args...; kws...))
    stats = cache_call(batchsolve, (solver, params), (; nthreads), cache_path, key; call, kws...)
    called && @debug "$call summary" elapsed=stats.time samples=length(params)
    return stats.value
end

rootsolve(args...; kws...) = solve(args...; kws...).u
function cache_rootsolve(f, a, b, p, alg, abstol, reltol, cache_path, key; call="rootsolve", kws...)
    cnt::Int = 0
    prob = IntervalNonlinearProblem((a,b), p) do x, p
        cnt += 1
        f(x, p)
    end
    stats = cache_call(rootsolve, (prob, alg), (; abstol, reltol), cache_path, key; call, kws...)
    cnt > 0 && @debug "$call summary" elapsed=stats.time samples=cnt
    return stats.value
end

const gcnt = fill(0)

function cache_benchmark(f, args...; call="benchmark", nsample=3, kws...)
    called::Bool = false
    bench_f = function (args...; kws...)
        called = true
        samples = map(1:nsample) do n
            @timed f(args...; kws...)
        end
        (; samples,
        min = samples[findmin(s -> s.time, samples)[2]],
        max = samples[findmax(s -> s.time, samples)[2]],
        avg = sum(s -> s.time, samples)/nsample,
        )
    end
    stats = cache_call(bench_f, args...; call, kws...)
    called && @debug "$call summary" elapsed=stats.time samples=nsample
    return stats.value
end
