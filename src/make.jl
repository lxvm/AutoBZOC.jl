function make(; figurepath=pwd(), kpathdensity=true, oc_fermiliquid=true, crossover=true, cfs=true, kws...)
    if kpathdensity
        f3a = fig3a(; kws...)
        save(joinpath(figurepath, "kpathdensity.png"), f3a)
    end

    if oc_fermiliquid
        f3 = fig3(; kws...)
        save(joinpath(figurepath, "oc_fermiliquid.png"), f3)
    end

    if crossover
        fb = fig_breakeven(; kws...)
        save(joinpath(figurepath, "crossover.png"), fb)
    end

    if cfs
        fc = fig_cfs(; kws...)
        save(joinpath(figurepath, "cfs.png"), fc)
    end

    return nothing
end

# make with single-precision turned on by default
make32(; kws...) = make(; nrtol=1e-4, σatol=1e-5u"Å^-1", kws..., prec=Float32)

function make_aux(; figurepath=pwd(), kpathdensity=true, oc_fermiliquid=true, crossover=true, kws...)
    if kpathdensity
        f3a = fig3a_aux(; kws...)
        save(joinpath(figurepath, "kpathdensity_aux.png"), f3a)
    end

    if oc_fermiliquid
        f3 = fig3_aux(; kws...)
        save(joinpath(figurepath, "oc_fermiliquid_aux.png"), f3)
    end

    if crossover
        fb = fig_breakeven_aux(; kws...)
        save(joinpath(figurepath, "crossover_aux.png"), fb)
    end

    return nothing
end

make32_aux(; kws...) = make_aux(; nrtol=1e-4, σatol=1e-5u"Å^-1", kws..., prec=Float32)
