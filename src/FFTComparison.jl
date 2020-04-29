using GeoStats
using SpectralGaussianSimulation
using SparseArrays
using Plots
using Random
using BlockArrays
using Distances
using StatsFuns
using JLD

include("./SortSparse.jl")
include("./KoLesky.jl")
include("./CovFuncs.jl")
include("./Utils.jl")

Random.seed!(123)

n = 1000
N = n^2
NTest = 20000
NTrain = N - NTest
NSamples = 1000

ρList = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8,0, 8.5, 9.0, 9.5, 10.0]
λ = 1.0
l = 0.5
covfunc!(r) = matern32!(r, l)
vario = MaternVariogram(range=l, order=1.5)
padFact = 5

# Prepare output Arrays
RMSE = zeros(length(ρList))
RMSE_int = zeros(length(ρList))
RMSE_exp = zeros(length(ρList))
coverage = zeros(length(ρList))
coverage_int = zeros(length(ρList))
coverage_exp = zeros(length(ρList))

# Sampling the process 
if (isfile("./out/jld/FFTData_n_$n.jld"))
  println("Loading FFT data from file")
  ld = load("./out/jld/FFTData_n_$n.jld")
  x = ld["x"]
  Y = ld["Y"]
else
  println("FFT data not found. Computing it from scratch")
  @time x, Y = specSampleGrid2D(n, NSamples, padFact, vario)
  save("./out/jld/FFTData_n_$n.jld", "x", x, "Y", Y)
end

# random shuffle of indices 
shuffleIDs = shuffle(1:N)
# Selecting groups of nodes as test data 
innerInds = [i for i in 1:N if (norm(x[:, i] .- 0.3) < 0.05 || norm((x[:,i] .- 0.75) .* [1.0, 0.1]) < 0.01)]
shuffleIDs = vcat(shuffleIDs, innerInds)
shuffleIDs = unique(shuffleIDs[end:-1:1])[end:-1:1]


# Creating data to be used by each run of for loop
#Reordering data 
xBase = x[:, shuffleIDs]
YBase = Y[shuffleIDs, :]
#Split data into train and testset
xTrainBase = xBase[:, 1 : NTrain]
xTestBase = xBase[:, (NTrain + 1) : end]


n_extrapolation = min(NTest, length(innerInds))
interpolation_indices_base = (NTrain + 1) : (NTrain + NTest - n_extrapolation)
extrapolation_indices_base = (NTrain + NTest - n_extrapolation + 1) : N

for (ρInd, ρ) in enumerate(ρList)
  x = copy(xBase)
  Y = copy(YBase)
  xTrain = copy(xTrainBase)
  xTest = copy(xTestBase)
  global RMSE
  global RMSE_int
  global RMSE_exp
  global coverage
  global coverage_int
  global coverage_exp

  colptr, rowval, P, revP, distances = sortSparseRev(xTrain, xTest, ρ, 1 )
  x = x[:, P]

  Y = Y[P, :]
  Y = PseudoBlockArray(Y, [NTest, NTrain], [NSamples])

  xTest = x[:, 1 : NTest]
  xTrain = x[:, (NTest + 1) : end]

  interpolation_indices = revP[interpolation_indices_base]
  extrapolation_indices = revP[extrapolation_indices_base]

  skeletons = construct_skeletons(colptr, rowval, distances, λ)
  @time L = assembleL(skeletons, x, covfunc!)
  L = PseudoBlockArray(L, [NTest, NTrain], [NTest, NTrain])

  # Compute posterior covariance 
  PΘApprox = Matrix(Hermitian(inv(Matrix(L[Block(1), Block(1)] * L[Block(1), Block(1)]'))))
  # Compute posterior mean
  μApprox = -L[Block(1), Block(1)]' \ (L[Block(2), Block(1)]' * Y[Block(2), :])

  # Computing coverage of the 0.9 confidence interval
  α = 0.1
  coverageMatrix = Float64.(abs.(μApprox - Y[Block(1), :]) ./ sqrt.(diag(PΘApprox)) .<= norminvcdf(1.0 - α/2))


  RMSE[ρInd] = sqrt(mean((μApprox .- Y[Block(1), :]).^2))
  RMSE_int[ρInd] = sqrt(mean(((μApprox .- Y[Block(1), :]).^2)[interpolation_indices, :]))
  RMSE_exp[ρInd] = sqrt(mean(((μApprox .- Y[Block(1), :]).^2)[extrapolation_indices, :]))

  coverage[ρInd] = mean(coverageMatrix)
  coverage_int[ρInd] = mean(coverageMatrix[interpolation_indices, :])
  coverage_exp[ρInd] = mean(coverageMatrix[extrapolation_indices, :])
end

save("./out/jld/FFTVaryRho_lambda_$(Int(10 * λ)).jld",
     "ρList", ρList, 
     "RMSE", RMSE,
     "RMSE_int", RMSE_int,
     "RMSE_exp", RMSE_exp,
     "coverage", coverage,
     "coverage_int", coverage_int,
     "coverage_exp", coverage_exp,
     "xTest", xTestBase)
