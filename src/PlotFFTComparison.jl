using JLD
using Plots
using LaTeXStrings
using Formatting
using TexTables
#reset past font scaling
Plots.scalefontsizes()
#scale fonts
Plots.scalefontsizes(1.75)
#Extracting stored varibles

lightblue = colorant"rgb(63%,74%,78%)"
orange = colorant"rgb(85%,55%,13%)"
silver = colorant"rgb(69%,67%,66%)"
rust = colorant"rgb(72%,26%,6%)"


ld = load("./out/jld/FFTVaryRho_lambda_10.jld")
for key in keys(ld)
  symb = Symbol(key)
  data = ld[key]
  @eval $(symb) = $(data)
end
accplot = plot(xlabel=L"\rho", ylabel=L"\log_{10}\left(\mathrm{RMSE}\right)", legend=:bottomright, ylims=(-4.25,-0.75), xlims=(0.5, 10.0), xticks=1:1:10)

covplot = plot(xlabel=L"\rho", ylabel=L"\mathrm{coverage}", legend=:bottomright, ylims=(0.72, 0.92), xlims=(0.5, 10.0), xticks=1:1:10)

hline!(covplot, [0.9], color=:black, label=L"\mathrm{theoretical}\  \mathrm{coverage}")

plot!(accplot,
      ρList,
      log10.(RMSE_int),
      label=L"\lambda = 1.0, \mathrm{scattered}",
      markershape=:circle,
      markersize=5,
      color=silver,
      linestyle=:solid,
      linewidth=5)

plot!(accplot,
      ρList,
      log10.(RMSE_exp),
      label=L"\lambda = 1.0, \mathrm{region}",
      markershape=:dtriangle,
      markersize=5,
      color=lightblue,
      linestyle=:solid,
      linewidth=5)

plot!(covplot,
      ρList,
      coverage_int,
      label=L"\lambda = 1.0, \mathrm{scattered}",
      markershape=:circle,
      markersize=5,
      color=silver,
      linestyle=:solid,
      linewidth=5)

plot!(covplot,
      ρList,
      coverage_exp,
      label=L"\lambda = 1.0, \mathrm{region}",
      markershape=:dtriangle,
      markersize=5,
      color=lightblue,
      linestyle=:solid,
      linewidth=5)



ld = load("./out/jld/FFTVaryRho_lambda_13.jld")
for key in keys(ld)
  symb = Symbol(key)
  data = ld[key]
  @eval $(symb) = $(data)
end

plot!(accplot,
      ρList,
      log10.(RMSE_int),
      label=L"\lambda = 1.3, \mathrm{scattered}",
      markershape=:circle,
      markersize=5,
      color=rust,
      linestyle=:dash,
      linewidth=5)

plot!(accplot,
      ρList,
      log10.(RMSE_exp),
      label=L"\lambda = 1.3, \mathrm{region}",
      markershape=:utriangle,
      markersize=5,
      color=orange,
      linestyle=:dot,
      linewidth=5)

plot!(covplot,
      ρList,
      coverage_int,
      label=L"\lambda = 1.3, \mathrm{scattered}",
      markershape=:circle,
      markersize=5,
      color=rust,
      linestyle=:dash,
      linewidth=5)

plot!(covplot,
      ρList,
      coverage_exp,
      label=L"\lambda = 1.3, \mathrm{region}",
      markershape=:utriangle,
      markersize=5,
      color=orange,
      linestyle=:dot,
      linewidth=5)

savefig(accplot, "./out/plots/FFT_accplot.pdf")
savefig(covplot, "./out/plots/FFT_covplot.pdf")

# # ----------------------------------------
# # Creating plot of test set
# # ----------------------------------------
# 
# xTestPlot = scatter(xTest[1,:], xTest[2,:],
#                     markersize=0.5,
#                     markerstrokewidth=0.1,
#                     legend=false)
# 
# savefig(xTestPlot, "./out/plots/FFTxTestScatter.png")


# # ----------------------------------------
# # Creating table
# # ----------------------------------------
# 
# outputTables = Array{IndexedTable{1,1},1}(undef, length(ρList))
# for (ρInd, ρ) in enumerate(ρList)
#   global outputTables
#   global coverage
#   global RMSE
#   RMSETab = TableCol("\$\\rho = $ρ\$",
#                      [Symbol("\$\\mathrm{RMSE}\$")],
#                      [FormattedNumber(RMSE[ρInd])]) 
# 
#   coverageTab = TableCol("\$\\rho = $ρ\$",
#                          [Symbol("\$\\mathrm{Coverage}\$")],
#                          [FormattedNumber(coverage[ρInd])]) 
#   outputTables[ρInd] = vcat(RMSETab, coverageTab)
# end
# 
# tab = hcat(outputTables[:]...)
# f = open("./out/tables/FFTTable.tex", "w")
# write(f, to_tex(tab))
# close(f)




     
