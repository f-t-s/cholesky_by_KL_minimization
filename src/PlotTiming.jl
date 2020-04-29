using JLD
using Plots
using LaTeXStrings
using Formatting
using TexTables
#reset past font scaling
Plots.scalefontsizes()
#scale fonts
Plots.scalefontsizes(1.75)

lightblue = colorant"rgb(63%,74%,78%)"
orange = colorant"rgb(85%,55%,13%)"
silver = colorant"rgb(69%,67%,66%)"
rust = colorant"rgb(72%,26%,6%)"

#Extracting stored varibles
ld = load("./out/jld/TimingVaryRho.jld")
for key in keys(ld)
  symb = Symbol(key)
  data = ld[key]
  @eval $(symb) = $(data)
end
# ----------------------------------------
# Creating plot of RMSE
# ----------------------------------------
plotTiming = plot(ylabel=L"$\mathrm{Time} \ \mathrm{in} \ \mathrm{s}$",
                  xlabel=L"\rho",
                  legend=:topleft
                  )

plotTimingVnonz = plot(ylabel=L"$\mathrm{Time} \ \mathrm{in} \ \mathrm{s}$",
                  xlabel=L"$\mathrm{nonzeros}$",
                  legend=false,
                  )


plotTimingBessel = plot(ylabel=L"$\mathrm{Time} \ \mathrm{in} \ \mathrm{s}$",
                        xlabel=L"\rho",
                        legend=:topleft,
                        )

plotTimingBesselVnonz = plot(ylabel=L"$\mathrm{Time} \ \mathrm{in} \ \mathrm{s}$",
                        xlabel=L"$\mathrm{nonzeros}$",
                        legend=false,
                        )



plot!(plotTiming,
      ρList,
      timings,
      markershape=:circle,
      color=orange,
      markersize=15,
      linewidth=5,
      linestyle=:dash,
      label="Exponential, no aggretation ")

plot!(plotTiming,
      ρList,
      timingsAgg,
      markershape=:diamond,
      color=lightblue,
      markersize=15,
      linewidth=5,
      linestyle=:dot,
      label="Exponential, with aggregation")

plot!(plotTimingVnonz,
      nonz,
      timings,
      markershape=:circle,
      color=orange,
      markersize=15,
      linewidth=5,
      linestyle=:dash,
      label="Exponential, no aggretation ")

plot!(plotTimingVnonz,
      nonzAgg,
      timingsAgg,
      markershape=:diamond,
      color=lightblue,
      markersize=15,
      linewidth=5,
      linestyle=:dot,
      label="Exponential, with aggregation")

plot!(plotTimingBessel,
      ρList,
      timingsBessel,
      markershape=:circle,
      color=rust,
      markersize=15,
      linewidth=5,
      linestyle=:dash,
      label="Bessel, no aggregation")

plot!(plotTimingBessel,
      ρList,
      timingsBesselAgg,
      markershape=:diamond,
      color=silver,
      markersize=15,
      linewidth=5,
      linestyle=:dot,
      label="Bessel, aggregation")

plot!(plotTimingBesselVnonz,
      nonzBessel,
      timingsBessel,
      markershape=:circle,
      color=rust,
      markersize=15,
      linewidth=5,
      linestyle=:dash,
      label="Bessel, no aggregation")

plot!(plotTimingBesselVnonz,
      nonzBesselAgg,
      timingsBesselAgg,
      markershape=:diamond,
      color=silver,
      markersize=15,
      linewidth=5,
      linestyle=:dot,
      label="Bessel, aggregation")



savefig(plotTiming, "./out/plots/timing.pdf")
savefig(plotTimingBessel, "./out/plots/timingBessel.pdf")

savefig(plotTimingVnonz, "./out/plots/timingVnonz.pdf")
savefig(plotTimingBesselVnonz, "./out/plots/timingBesselVnonz.pdf")