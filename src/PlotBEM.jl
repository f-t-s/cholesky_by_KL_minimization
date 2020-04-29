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

#Extracting stored varibles
ld = load("./out/jld/BEM_Plot.jld")
for key in keys(ld)
  symb = Symbol(key)
  data = ld[key]
  @eval $(symb) = $(data)
end
accplot = plot(xlabel=L"$q$", 
               ylabel=L"$\log_{10}(\mathrm{RMSE})$",
               # legend=:bottomleft,
               ylim=(-2.5,0.5)) 
plot!(accplot,
      (3 : 8),
      log10.(sqrt.(accuracy_exact)),
      label=L"\mathrm{dense}",
      markershape=:circle,
      markersize=15,
      color=silver,
      linestyle=:solid,
      linewidth=5)

plot!(accplot,
      3 : 8,
      log10.(sqrt.(accuracy_predict[:, 1])),
      label=L"\rho = 1.0",
      markershape=:dtriangle,
      markersize=15,
      color=rust,
      linestyle=:dash,
      linewidth=5)

plot!(accplot,
      3 : 8,
      log10.(sqrt.(accuracy_predict[:, 2])),
      label=L"\rho = 2.0",
      markershape=:utriangle,
      markersize=15,
      color=orange,
      linestyle=:dash,
      linewidth=5)

plot!(accplot,
      3 : 8,
      log10.(sqrt.(accuracy_predict[:, 3])),
      label=L"\rho = 3.0",
      markershape=:diamond,
      markersize=15,
      color=:lightblue,
      linestyle=:dot,
      linewidth=5)

savefig(accplot, "./out/plots/BEM_accplot.pdf")

nnzplot = plot(xlabel=L"$q$", 
               ylabel=L"$\log_{10}\left(\mathrm{Nonzeros}\right)$",
               legend=:topleft,
               #ylim=(-2.5,0.5)
               ) 
plot!(nnzplot,
      (3 : 8),
      log10.((nnz_exact)),
      label=L"\mathrm{dense}",
      markershape=:circle,
      markersize=15,
      color=silver,
      linestyle=:solid,
      linewidth=5)

plot!(nnzplot,
      3 : 8,
      log10.((nnz_predict[:, 1])),
      label=L"\rho = 1.0",
      markershape=:dtriangle,
      markersize=15,
      color=rust,
      linestyle=:dash,
      linewidth=5)

plot!(nnzplot,
      3 : 8,
      log10.((nnz_predict[:, 2])),
      label=L"\rho = 2.0",
      markershape=:utriangle,
      markersize=15,
      color=orange,
      linestyle=:dash,
      linewidth=5)

plot!(nnzplot,
      3 : 8,
      log10.((nnz_predict[:, 3])),
      label=L"\rho = 3.0",
      markershape=:diamond,
      markersize=15,
      color=:lightblue,
      linestyle=:dot,
      linewidth=5)

savefig(nnzplot, "./out/plots/BEM_nnzplot.pdf")

timeplot = plot(xlabel=L"$q$", 
                ylabel=L"$\log_{10}\left(\mathrm{time}\ \mathrm{in}\  \mathrm{seconds}\right)$",
                # legend=:bottomleft,
                #ylim=(-2.5,0.5)
                ) 
plot!(timeplot,
      (3 : 8),
      log10.((timing_exact)),
      label=L"\mathrm{dense}",
      markershape=:circle,
      markersize=15,
      color=silver,
      linestyle=:solid,
      linewidth=5)

plot!(timeplot,
      3 : 8,
      log10.(sqrt.(timing_predict[:, 1])),
      label=L"\rho = 1.0",
      markershape=:dtriangle,
      markersize=15,
      color=rust,
      linestyle=:dash,
      linewidth=5)

plot!(timeplot,
      3 : 8,
      log10.(sqrt.(timing_predict[:, 2])),
      label=L"\rho = 2.0",
      markershape=:utriangle,
      markersize=15,
      color=orange,
      linestyle=:dash,
      linewidth=5)

plot!(timeplot,
      3 : 8,
      log10.(sqrt.(timing_predict[:, 3])),
      label=L"\rho = 3.0",
      markershape=:diamond,
      markersize=15,
      color=:lightblue,
      linestyle=:dot,
      linewidth=5)

savefig(timeplot, "./out/plots/BEM_timeplot.pdf")
