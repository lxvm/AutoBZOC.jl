function make(; figurepath=pwd(), kws...)
    f3a = fig3a(; kws...)
    save(joinpath(figurepath, "kpathdensity.png"), f3a)

    f3 = fig3(; kws...)
    save(joinpath(figurepath, "oc_fermiliquid.png"), f3)

    return nothing
end

# make with single-precision turned on by default
make32(; kws...) = make(; nrtol=1e-4, σatol=1e-5u"Å^-1", prec=Float32, kws...)

function make_aux(; figurepath=pwd(), kws...)
    f3a = fig3a_aux(; kws...)
    save(joinpath(figurepath, "kpathdensity_aux.png"), f3a)

    f3 = fig3_aux(; kws...)
    save(joinpath(figurepath, "oc_fermiliquid_aux.png"), f3)

    return nothing
end

make32_aux(; kws...) = make_aux(; nrtol=1e-4, σatol=1e-5u"Å^-1", prec=Float32, kws...)
