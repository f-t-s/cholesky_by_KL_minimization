using JLD
using Plots

using LaTeXStrings
#reset past font scaling
Plots.scalefontsizes()
#scale fonts
Plots.scalefontsizes(1.75)

lightblue = colorant"rgb(63%,74%,78%)"
orange = colorant"rgb(85%,55%,13%)"
silver = colorant"rgb(69%,67%,66%)"
rust = colorant"rgb(72%,26%,6%)"

linestyle_list =[:dash, :dashdotdot, :dot]
markershape_list=[:utriangle, :circle, :dtriangle]
label_list =["\\nu = 1/2", "\\nu = 3/2" , "\\nu = 5/2"]

meanplot_combined = plot(xlabel=L"$\rho$",
                         ylabel=L"$\log_{10}(\mathrm{RMSE}\left(\mu\right))$",
                         legend=false, 
                         )

stdplot_combined = plot(xlabel=L"$\rho$",
                        ylabel=L"$\log_{10}(\mathrm{RMSE}\left(\sigma\right))$",
                        legend=false, 
                        )



for (maternIndex, maternOrder) in enumerate(["12", "32", "52"])
  global meanplot_combined
  global stdplot_combined
  #Extracting stored varibles
  ld = load("./out/jld/IncludePredFarVaryRho" * maternOrder * ".jld")
  for key in keys(ld)
    symb = Symbol(key)
    data = ld[key]
    @eval $(symb) = $(data)
  end

  # meanPlot
  # changed convention of what to call first and last in paper
  plot!(meanplot_combined, ρList, 
        log10.(meanRMSENoPred), 
        markersize=15,
        linewidth=5,
        color=silver,
        linestyle=linestyle_list[maternIndex],
        markershape=markershape_list[maternIndex])

  # changed convention of what to call first and last in paper
  plot!(meanplot_combined, ρList,
        log10.(meanRMSEPredLast),
        markersize=15,
        linewidth=5,
        color=orange,
        linestyle=linestyle_list[maternIndex],
        markershape=markershape_list[maternIndex])

  # changed convention of what to call first and last in paper
  plot!(meanplot_combined, ρList,
        log10.(meanRMSEPredFirst),
        markersize=15,
        linewidth=5,
        color=lightblue,
        linestyle=linestyle_list[maternIndex],
        markershape=markershape_list[maternIndex])

  # STD plot
  # changed convention of what to call first and last in paper
  plot!(stdplot_combined, ρList, 
        log10.(stdRMSENoPred), 
        markersize=15,
        linewidth=5,
        color=silver,
        linestyle=linestyle_list[maternIndex],
        markershape=markershape_list[maternIndex])

  # changed convention of what to call first and last in paper
  plot!(stdplot_combined, ρList,
      log10.(stdRMSEPredLast),
      markersize=15,
      linewidth=5,
      color=orange,
      linestyle=linestyle_list[maternIndex],
      markershape=markershape_list[maternIndex])

  # changed convention of what to call first and last in paper
  plot!(stdplot_combined, ρList,
        log10.(stdRMSEPredFirst),
        markersize=15,
        linewidth=5,
        color=lightblue,
        linestyle=linestyle_list[maternIndex],
        markershape=markershape_list[maternIndex])
end

  savefig(meanplot_combined, "./out/plots/predFarMean_combined.pdf")
  savefig(stdplot_combined, "./out/plots/predFarSTD_combined.pdf")

for (maternIndex, maternOrder) in enumerate(["12", "32", "52"])
      legendplot = plot(grid =false, showaxis=false, size=(450, 150), framestyle=:none, legend=:left)
      plot!(legendplot,
            [],
            label=LaTeXString("\$\\mathrm{no} \\ \\mathrm{predictions}, \\ " * label_list[maternIndex] * "\$"),
            markersize=15,
            linewidth=5,
            color=silver,
            linestyle=linestyle_list[maternIndex],
            markershape=markershape_list[maternIndex])

      plot!(legendplot,
            [],
            label=LaTeXString("\$\\mathrm{predictions} \\ \\mathrm{first}, \\ " * label_list[maternIndex] * "\$"),
            markersize=15,
            linewidth=5,
            color=orange,
            linestyle=linestyle_list[maternIndex],
            markershape=markershape_list[maternIndex])

      plot!(legendplot,
            [],
            label=LaTeXString("\$\\mathrm{predictions} \\ \\mathrm{last}, \\ " * label_list[maternIndex] * "\$"),
            markersize=15,
            linewidth=5,
            color=lightblue,
            linestyle=linestyle_list[maternIndex],
            markershape=markershape_list[maternIndex])
      savefig(legendplot, "./out/plots/legend_" * maternOrder * ".pdf")
end

# Plotting the measurement points: 
ld = load("./out/jld/IncludePredFarVaryRho12.jld")
for key in keys(ld)
  symb = Symbol(key)
  data = ld[key]
  @eval $(symb) = $(data)
end

xplot = plot(size=(400,400), aspect_ratio=:equal, xticks=false, yticks=false, axis=false, xlims=(-1.1,3.1), ylims=(-1.1,3.1), legend=:topright)
scatter!(xplot, vec(xTrain[1,1:20:end]), vec(xTrain[2,1:20:end]), color=silver, label=L"\mathrm{training}", markerstrokecolor=silver, markersize=5.0, markerstrokewidth = 0.0)
scatter!(xplot, vec(xTest[1,:]), vec(xTest[2,:]), color=rust, label=L"\mathrm{prediction}", markerstrokewidth=0.0, markersize=5.0)
savefig(xplot, "./out/plots/PointsFar.pdf")

# cropping legend
run(`pdfcrop --margins '5 5 5 5' ./out/plots/legend_12.pdf ./out/plots/legend_12.pdf`)
run(`pdfcrop --margins '5 5 5 5' ./out/plots/legend_32.pdf ./out/plots/legend_32.pdf`)
run(`pdfcrop --margins '5 5 5 5' ./out/plots/legend_52.pdf ./out/plots/legend_52.pdf`) 