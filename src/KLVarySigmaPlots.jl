using JLD
using Plots
using Colors 
using LaTeXStrings

#reset past font scaling
Plots.scalefontsizes()
#scale fonts
Plots.scalefontsizes(1.75)

lightblue = colorant"rgb(63%,74%,78%)"
orange = colorant"rgb(85%,55%,13%)"
silver = colorant"rgb(69%,67%,66%)"
rust = colorant"rgb(72%,26%,6%)"

#Extracting stored varibles
ld = load("./out/jld/KLVarySigma.jld")
for key in keys(ld)
  symb = Symbol(key)
  data = ld[key]
  @eval $(symb) = $(data)
end
for (i, sm) in enumerate(["12", "32", "52"])

  newplot = plot(xlabel=L"$\log_{10}{\sigma}$", 
                 ylabel=L"$\log_{10}(\mathrm{symmetrized} \ \mathrm{KL})$",
                 legend=:bottomleft,
                 ylim=(-6,4)) 

  plot!(newplot,
        log10.(σList),
        log10.(KLTrueNaive[i, :] + KLNaiveTrue[i, :]),
        label=L"\mathrm{Naive}",
        markershape=:diamond,
        markersize=15,
        color=lightblue,
        linestyle=:dash,
        linewidth=5)

  plot!(newplot,
        log10.(σList),
        log10.(KLTrueApp[i, :] + KLAppTrue[i, :]),
        label=L"\mathrm{IC}, \mathrm{nonzeros}(L)",
        markershape=:dtriangle,
        markersize=15,
        color=orange,
        linestyle=:dash,
        linewidth=5)

  plot!(newplot,
        log10.(σList),
        log10.(KLTrueLarge[i, :] + KLLargeTrue[i, :]),
        label=L"\mathrm{IC}, \mathrm{nonzeros}(L L^{\top})",
        markershape=:utriangle,
        markersize=15,
        color=rust,
        linestyle=:dash,
        linewidth=5)

  plot!(newplot,
        log10.(σList),
        log10.(KLTrueExact[i, :] + KLExactTrue[i, :]),
        label=L"\mathrm{Exact}",
        markershape=:circle,
        markersize=15,
        color=silver,
        linestyle=:dot,
        linewidth=5)

  savefig(newplot, "./out/plots/KLVarySigma$(sm).pdf")
  display(newplot)
end