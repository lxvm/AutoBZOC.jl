function fig_bands(; kws...)
    (; model, kpath, N_k, T) = merge(default, NamedTuple(kws))

    μ, = findchempot(; kws..., T)
    h, = model(; kws...)
    AutoBZ.shift!(h, μ)


    kp = kpath(; kws...)
    kpi = interpolate(kp, N_k)
    kps = KPathSegment(kpi.basis, only(kpi.kpaths), only(kpi.labels), kpi.setting)
    kloc = cumdists(kps.kpath)
    ticks_k = (kloc[collect(keys(kps.label))], map(string, values(kps.label)))

    fig = Figure()
    ax = Axis(fig[1,1];
        xlabel="",
        ylabel="",
        xticks=ticks_k,
    )
    kpathinterpplot!(ax, kps, h)

    return fig
end