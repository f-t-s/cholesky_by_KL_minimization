using JLD
using Plots
using LaTeXStrings
using TexTables
import Base.Iterators.product
import Distributions.mean
import Distributions.std
#reset past font scaling
Plots.scalefontsizes()
#scale fonts
Plots.scalefontsizes(1.5)
#Extracting stored varibles
for (maternOrder, dset) in product(["12", "32"], ["Far", "Close", "Circle"])
  ld = load("./out/jld/IncludePred$(dset)$(maternOrder).jld")
  for key in keys(ld)
    symb = Symbol(key * dset * maternOrder)
    data = ld[key]
    @eval $(symb) = $(data)
  end
end

# --------------------
# Start out by creating plots of the data 
# --------------------
NTrain = 10000
NTest = 10
d = 2
N = NTrain + NTest

# Prediction points close
xTrain = rand(d, NTrain)
xTest = rand(d, NTest)

plotClose = plot(xlims=(0.0,1.5),
                 ylims=(0.0,1.5),
                 legend=:topleft)
scatter!(plotClose,
         xTrain[1, :],
         xTrain[2, :],
         markershape=:circle,
         color=:blue,
         label=L"\mathrm{Training Points}")

scatter!(plotClose,
         xTest[1, :],
         xTest[2, :],
         markershape=:diamond,
         color=:orange,
         markersize=7.0,
         label=L"\mathrm{Prediction Points}")
savefig(plotClose, "./out/plots/predClose.pdf")

# Prediction points far
xTrain = rand(d, NTrain)
xTest = 1.1 * ones(d,NTest) .+ 0.1 * rand(d, NTest)

plotFar = plot(xlims=(0.0,1.5),
               ylims=(0.0,1.5),
               legend=:topleft)
scatter!(plotFar,
         xTrain[1, :],
         xTrain[2, :],
         markershape=:circle,
         color=:blue,
         label=L"\mathrm{Training Points}")

scatter!(plotFar,
         xTest[1, :],
         xTest[2, :],
         markershape=:diamond,
         color=:orange,
         markersize=7.0,
         label=L"\mathrm{Prediction Points}")
savefig(plotFar, "./out/plots/predFar.pdf")

h = 1 / NTrain
xTrain = h : h : 1 
xTrain = vcat(sin.(2 * π * xTrain)', 
              2 * cos.(2 * π * xTrain)')
xTest = 0.1 * randn(2,NTest) 

# Reducing the size of training set to improve visualization
NTrain = NTrain / 20
# Prediction points circle
h = 1 / NTrain
xTrain = h : h : 1 
xTrain = vcat(sin.(2 * π * xTrain)', 
              2 * cos.(2 * π * xTrain)')
xTest = 0.1 * randn(2,NTest) 

plotCircle = plot(legend=:topleft)
scatter!(plotCircle,
         xTrain[1, :],
         xTrain[2, :],
         markershape=:circle,
         color=:blue,
         label=L"\mathrm{Training\ Points}")

scatter!(plotCircle,
         xTest[1, :],
         xTest[2, :],
         markershape=:diamond,
         color=:orange,
         markersize=7.0,
         label=L"\mathrm{Prediction\ Points}")
savefig(plotCircle, "./out/plots/predCircle.pdf")

# --------------------
# Create tables 
# --------------------
outputTables = Array{IndexedTable{1,1},1}(undef, 6)

for (k, (dset, maternOrder)) in enumerate(
                                product(["Close", "Far", "Circle"], ["12", "32"]))
  

  @eval meanErrorNoPred = $(Symbol("meanErrorNoPred" * dset * maternOrder))
  @eval stdErrorNoPred = $(Symbol("stdErrorNoPred" * dset * maternOrder))
  @eval KLTrueApproxNoPred = $(Symbol("KLTrueApproxNoPred" * dset * maternOrder))
  @eval KLApproxTrueNoPred = $(Symbol("KLApproxTrueNoPred" * dset * maternOrder))
  @eval meanErrorPredFirst = $(Symbol("meanErrorPredFirst" * dset * maternOrder))
  @eval stdErrorPredFirst = $(Symbol("stdErrorPredFirst" * dset * maternOrder))
  @eval KLTrueApproxPredFirst = $(Symbol("KLTrueApproxPredFirst" * dset * maternOrder))
  @eval KLApproxTruePredFirst = $(Symbol("KLApproxTruePredFirst" * dset * maternOrder))
  @eval meanErrorPredLast = $(Symbol("meanErrorPredLast" * dset * maternOrder))
  @eval stdErrorPredLast = $(Symbol("stdErrorPredLast" * dset * maternOrder))
  @eval KLTrueApproxPredLast = $(Symbol("KLTrueApproxPredLast" * dset * maternOrder))
  @eval KLApproxTruePredLast = $(Symbol("KLApproxTruePredLast" * dset * maternOrder))


  meanTabNoPred = TableCol("\$\\mathrm{$(dset)}, \\nu = $(maternOrder[1]) / $(maternOrder[2])\$",
                     [Symbol("\$\\left|\\mu-\\mu_{\\NP}\\right|\$")],
                     [mean(meanErrorNoPred)],
                     [std(meanErrorNoPred)],
                     )

  meanTabPredFirst = TableCol("\$\\mathrm{$(dset)}, \\nu = $(maternOrder[1]) / $(maternOrder[2])\$",
                     [Symbol("\$\\left|\\mu-\\mu_{\\PF}\\right|\$")],
                     [mean(meanErrorPredFirst)],
                     [std(meanErrorPredFirst)],
                     )

  meanTabPredLast = TableCol("\$\\mathrm{$(dset)}, \\nu = $(maternOrder[1]) / $(maternOrder[2])\$",
                     [Symbol("\$\\left|\\mu-\\mu_{\\PL}\\right|\$")],
                     [mean(meanErrorPredLast)],
                     [std(meanErrorPredLast)],
                     )

  stdTabNoPred = TableCol("\$\\mathrm{$(dset)}, \\nu = $(maternOrder[1]) / $(maternOrder[2])\$",
                       [Symbol(:"\$\\left|\\sigma^2-\\sigma_{\\NP}^2\\right|\$")],
                       [mean(stdErrorNoPred)],
                       # [std(stdErrorNoPred)],
                       )

  stdTabPredFirst = TableCol("\$\\mathrm{$(dset)}, \\nu = $(maternOrder[1]) / $(maternOrder[2])\$",
                       [Symbol("\$\\left|\\sigma^2-\\sigma_{\\PF}^2\\right|\$")],
                       [mean(stdErrorPredFirst)],
                       # [std(stdErrorPredFirst)],
                       )

  stdTabPredLast = TableCol("\$\\mathrm{$(dset)}, \\nu = $(maternOrder[1]) / $(maternOrder[2])\$",
                       [Symbol("\$\\left|\\sigma^2-\\sigma_{\\PL}^2\\right|\$")],
                       [mean(stdErrorPredLast)],
                       # [std(stdErrorPredLast)],
                       )

  KLTrueAppTabNoPred = TableCol("\$\\mathrm{$(dset)}, \\nu = $(maternOrder[1]) / $(maternOrder[2])\$",
                     [Symbol("\$\\KL*{\\KM}{\\KM_{\\NP}}\$")],
                     [mean(KLTrueApproxNoPred)],
                     # [std(KLTrueApproxNoPred)],
                     )

  KLTrueAppTabPredFirst = TableCol("\$\\mathrm{$(dset)}, \\nu = $(maternOrder[1]) / $(maternOrder[2])\$",
                     [Symbol("\$\\KL*{\\KM}{\\KM_{\\PF}}\$")],
                     [mean(KLTrueApproxPredFirst)],
                     # [std(KLTrueApproxPredFirst)],
                     )

  KLTrueAppTabPredLast = TableCol("\$\\mathrm{$(dset)}, \\nu = $(maternOrder[1]) / $(maternOrder[2])\$",
                     [Symbol("\$\\KL*{\\KM}{\\KM_{\\PL}}\$")],
                     [mean(KLTrueApproxPredLast)],
                     # [std(KLTrueApproxPredLast)],
                     )

  KLAppTrueTabNoPred = TableCol("\$\\mathrm{$(dset)}, \\nu = $(maternOrder[1]) / $(maternOrder[2])\$",
                     [Symbol("\$\\KL*{\\KM_{\\NP}}{\\KM}\$")],
                     [mean(KLApproxTrueNoPred)],
                     # [std(KLApproxTrueNoPred)],
                     )

  KLAppTrueTabPredFirst = TableCol("\$\\mathrm{$(dset)}, \\nu = $(maternOrder[1]) / $(maternOrder[2])\$",
                     [Symbol("\$\\KL*{\\KM_{\\PF}}{\\KM}\$")],
                     [mean(KLApproxTruePredFirst)],
                     # [std(KLApproxTruePredFirst)],
                     )

  KLAppTrueTabPredLast = TableCol("\$\\mathrm{$(dset)}, \\nu = $(maternOrder[1]) / $(maternOrder[2])\$",
                     [Symbol("\$\\KL*{\\KM_{\\PL}}{\\KM}\$")],
                     [mean(KLApproxTruePredLast)],
                     # [std(KLApproxTruePredLast)],
                     )

  # Setting the format 

  setfield!.(meanTabNoPred.data.vals, :format, "{:,2e}")
  setfield!.(meanTabNoPred.data.vals, :format_se, "{:,2e}")

  setfield!.(meanTabPredFirst.data.vals, :format, "{:,2e}")
  setfield!.(meanTabPredFirst.data.vals, :format_se, "{:,2e}")

  setfield!.(meanTabPredLast.data.vals, :format, "{:,2e}")
  setfield!.(meanTabPredLast.data.vals, :format_se, "{:,2e}")

  setfield!.(stdTabNoPred.data.vals, :format, "{:,2e}")
  # setfield!.(stdTabNoPred.data.vals, :format_se, "{:,2e}")

  setfield!.(stdTabPredFirst.data.vals, :format, "{:,2e}")
  # setfield!.(stdTabPredFirst.data.vals, :format_se, "{:,2e}")

  setfield!.(stdTabPredLast.data.vals, :format, "{:,2e}")
  # setfield!.(stdTabPredLast.data.vals, :format_se, "{:,2e}")

  setfield!.(KLTrueAppTabNoPred.data.vals, :format, "{:,2e}")
  # setfield!.(KLTrueAppTabNoPred.data.vals, :format_se, "{:,2e}")

  setfield!.(KLTrueAppTabPredFirst.data.vals, :format, "{:,2e}")
  # setfield!.(KLTrueAppTabPredFirst.data.vals, :format_se, "{:,2e}")

  setfield!.(KLTrueAppTabPredLast.data.vals, :format, "{:,2e}")
  # setfield!.(KLTrueAppTabPredLast.data.vals, :format_se, "{:,2e}")

  setfield!.(KLAppTrueTabNoPred.data.vals, :format, "{:,2e}")
  # setfield!.(KLAppTrueTabNoPred.data.vals, :format_se, "{:,2e}")

  setfield!.(KLAppTrueTabPredFirst.data.vals, :format, "{:,2e}")
  # setfield!.(KLAppTrueTabPredFirst.data.vals, :format_se, "{:,2e}")

  setfield!.(KLAppTrueTabPredLast.data.vals, :format, "{:,2e}")
  # setfield!.(KLAppTrueTabPredLast.data.vals, :format_se, "{:,2e}")

  outputTables[k] = vcat(meanTabNoPred,
                         meanTabPredFirst,
                         meanTabPredLast,
                         stdTabNoPred,
                         stdTabPredFirst,
                         stdTabPredLast,
                         KLTrueAppTabNoPred,
                         KLTrueAppTabPredFirst,
                         KLTrueAppTabPredLast,
                         KLAppTrueTabNoPred,
                         KLAppTrueTabPredFirst,
                         KLAppTrueTabPredLast)
end
tab = hcat(outputTables[:]...)
f = open("./out/tables/IncludePredTables.tex", "w")
write(f, to_tex(tab))
close(f)


