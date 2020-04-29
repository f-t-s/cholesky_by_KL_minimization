using Random
using Plots
using SparseArrays
using GeoStats
using SpectralGaussianSimulation
using LaTeXStrings
#reset past font scaling
Plots.scalefontsizes()
#scale fonts
Plots.scalefontsizes(1.75)
#Extracting stored varibles

lightblue = colorant"rgb(63%,74%,78%)"
orange = colorant"rgb(85%,55%,13%)"
silver = colorant"rgb(69%,67%,66%)"
rust = colorant"rgb(72%,26%,6%)"

include("./CovFuncs.jl")
include("./Utils.jl")

Random.seed!(123)

n = 1000
N = n^2
NTest = 20000
NTrain = N - NTest

l = 0.1
covfunc!(r) = matern32!(r, l)
vario = MaternVariogram(range=l, order=1.5)
x, Y = specSampleGrid2D(n, 1, 1, vario)


α = 0.02
β = 0.02


# random shuffle of indices 
shuffleIDs = shuffle(1:N)
# Selecting groups of nodes as test data 
innerInds = [i for i in 1:N if (norm(x[:, i] .- 0.3) < 0.05 || norm((x[:,i] .- 0.75) .* [1.0, 0.1]) < 0.01)]
shuffleIDs = vcat(shuffleIDs, innerInds)
shuffleIDs = unique(shuffleIDs[end:-1:1])[end:-1:1]


# Creating data to be used by each run of for loop
#Reordering data 
xBase = x[:, shuffleIDs]

n_extrapolation = min(NTest, length(innerInds))
interpolation_indices_base = (NTrain + 1) : (NTrain + NTest - n_extrapolation)
extrapolation_indices_base = (NTrain + NTest - n_extrapolation + 1) : N

x_train = xBase[:, rand(1 : NTrain, round(Int, α * NTrain))] 
x_scattered = xBase[:, interpolation_indices_base[rand(1 : length(interpolation_indices_base), round(Int, β * length(interpolation_indices_base)))]]  
x_region = xBase[:, extrapolation_indices_base[rand(1 : length(extrapolation_indices_base), round(Int, β * length(extrapolation_indices_base)))]]  


outplot = plot(size=(400,400), aspect_ratio=:equal, xticks=false, yticks=false, axis=false, legend=:bottomright)
scatter!(outplot, vec(x_train[1,:]), vec(x_train[2,:]), color=silver, label=L"\mathrm{training}", markerstrokewidth=0.0, markeralpha=0.5)
scatter!(outplot, vec(x_scattered[1,:]), vec(x_scattered[2,:]), color=rust, label=L"\mathrm{scattered}", markerstrokewidth=0.0)
scatter!(outplot, vec(x_region[1,:]), vec(x_region[2,:]), color=orange, label=L"\mathrm{region}", markerstrokewidth=0.0)

savefig(outplot, "./out/plots/FFTPoints.pdf")


