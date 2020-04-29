using JLD
using Plots
using LaTeXStrings
using Colors
#reset past font scaling
Plots.scalefontsizes()
#scale fonts
Plots.scalefontsizes(1.75)


orange = colorant"rgb(85%,55%,13%)"

#Extracting stored varibles
ld = load("./out/jld/TestCGConvergence.jld")
for key in keys(ld)
  symb = Symbol(key)
  data = ld[key]
  @eval $(symb) = $(data)
end

outPlot = plot(xlabel=L"\mathrm{Iterations}", 
            ylabel=L"\log_{10}(\max_{\rho, \sigma, \nu} \|\mu - \mu_{\mathrm{app}}\| / \|\mu\|)") 

plot!(outPlot, log10.(maxError), legend=false, linestyle=:dash, markershape=:dtriangle, markersize=15, linewidth=5, color=orange)

savefig(outPlot, "./out/plots/TestCGConvergence.pdf")