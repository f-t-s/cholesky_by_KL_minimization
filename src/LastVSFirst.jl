include("SortSparse.jl")
include("KoLesky.jl")
include("Cholesky.jl")
include("CovFuncs.jl")
include("Utils.jl")
using LinearAlgebra
using Distributions
using Distances
using Random
using Match
using JLD
using BlockArrays
# Random.seed!(123)

NTrain = 20000
NTest = 100
N = NTrain + NTest
d = 2
l = 0.5
# For now, only properly implemented for $σ = 0$.
σ = 0.0 * exp.(randn(N))
# σ = 1e-2 * ones(N)
ρList = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
λ = 1.5
ρ = 3.0
maternOrder = "32"

#Creating arrays to hold the results

meanErrorClosePredFirst = zeros(length(ρList))
medianErrorClosePredFirst = zeros(length(ρList))
meanErrorClosePredLast = zeros(length(ρList))
medianErrorClosePredLast = zeros(length(ρList))
meanErrorFarPredFirst = zeros(length(ρList))
medianErrorFarPredFirst = zeros(length(ρList))
meanErrorFarPredLast = zeros(length(ρList))
medianErrorFarPredLast = zeros(length(ρList))


# Creating uniform random data
xTrain = rand(d, NTrain)
xTestClose = rand(d, NTest)
xTestFar = 1.1 * ones(d,NTest) .+ 0.1 * rand(d, NTest)

σ .= max.(σ, eps(Float64))
σ = PseudoBlockArray(σ, [NTest, NTrain])

covfunc12!(r) = matern12!(r,l)
covfunc32!(r) = matern32!(r,l)
covfunc52!(r) = matern52!(r,l)

# Make sure to avoid NANs
@match maternOrder begin
  "12" => (covfunc! = covfunc12!)
  "32" => (covfunc! = covfunc32!)
  "52" => (covfunc! = covfunc52!)
end
  
# =====================================
# Starting with interpolation:
# =====================================
# xTest = xTestClose
# xTest = xTestFar
# x = hcat(xTrain, xTest)

# =====================================
# Screw that, try different dataset:
# =====================================
h = 1 / NTrain
xTrain = h : h : 1 
# xTrain = rand(NTrain)
xTrain = vcat(sin.(2 * π * xTrain)', 
              2 * cos.(2 * π * xTrain)') # + 0.1 * randn(2, NTrain)
xTest = 0.1 * randn(2, NTest)
x = hcat(xTrain, xTest)


# Creating reference solution 
Θ = covfunc!(pairwise(Euclidean(), x, dims=2)) + Diagonal(σ)
L_Θ = cholesky(Matrix(Θ))
y = L_Θ.L * randn(N)
Θ = Θ + Diagonal(σ)

# Creating data
Θ = PseudoBlockArray(Θ, [NTrain, NTest], [NTrain, NTest])
y = PseudoBlockArray(y, [NTrain, NTest])
y[Block(1)] .+= Diagonal(sqrt.(σ[Block(2)])) * randn(NTrain)

# Computing the true posterior mean
μ = Θ[Block(2,1)] * (Θ[Block(1,1)] \ y[Block(1)])
Σ = Θ[Block(2,2)] - Θ[Block(2,1)] * (Θ[Block(1,1)] \ Θ[Block(1,2)])


# for (iρ, ρ) in enumerate(ρList)

  # =====================================
  # Computing posterior mean for prediction variables last
  # =====================================

  # reordering points and forming skeletons
  colptr, rowval, P, revP, distances = sortSparseRev(xTrain, xTest, ρ, 1)
  xOrd = x[:, P]
  xOrdTest = xOrd[:, 1 : NTest]
  xOrdTrain = xOrd[:, (NTest + 1) : end]
  skeletons = construct_skeletons(colptr, rowval, distances, λ)
  μOrd = μ[P[1 : NTest] .- NTrain]
  ΣOrd = Σ[P[1 : NTest] .- NTrain, P[1 : NTest] .- NTrain]
  yOrd = PseudoBlockArray(y[P], [NTest, NTrain])
  σOrd = PseudoBlockArray(σ[P], [NTest, NTrain])

  # Creating vecchia approximation
  L = assembleL(skeletons, xOrd, covfunc!)
  L22 = L[(NTest + 1) : end, (NTest + 1) : end]
  U22 = squareSparse(L22) + spdiagm(0 => 1 ./ σOrd[Block(2)])
  icholU_high_level!(U22)
  noisecov = NoiseCov(L22, U22, σOrd[Block(2)])
  μPredLast = (L' \ ( L \ vcat(zero(y[Block(2)]), noisecov \ (yOrd[Block(2)]))))[1:NTest]

  ΣPredLast = inv(Matrix(L[1 : NTest, 1 : NTest] * L[1 : NTest, 1 : NTest]'))

  # @show norm(μOrd - μPredLast) / norm(μOrd)
  # @show norm(μOrd - μPredLast)
  # @show norm(diag(ΣOrd) - diag(ΣPredLast))
  @show mean(abs.(μOrd - μPredLast))
  @show mean(abs.(diag(ΣOrd) - diag(ΣPredLast)))

  # =====================================
  # Computing posterior mean for prediction variables first
  # =====================================

  # reordering points and forming skeletons
  colptr, rowval, P, revP, distances = sortSparseRev(xTrain, ρ, 1)
  xOrdTrain = xTrain[:, P]
  xOrd = hcat(xOrdTrain, xTest)
  skeletons = construct_skeletons(colptr, rowval, distances, λ)
  μOrd = μ
  ΣOrd = Σ
  yOrd = y[Block(1)][P]
  σOrd = σ[P]

  μPredFirst, ΣPredFirst = predict(skeletons, (NTrain+1): N, yOrd, xOrd, covfunc!, σ)

  # @show norm(μOrd - μPredFirst)
  # @show norm(diag(ΣOrd) - ΣPredFirst)
  @show mean(abs.(μOrd - μPredFirst))
  @show mean(abs.(diag(ΣOrd) - ΣPredFirst))

  # =====================================
  # Don't include prediction variables at all
  # =====================================

  # reordering points and forming skeletons
  colptr, rowval, P, revP, distances = sortSparseRev(xTrain, ρ, 1)
  xOrdTrain = xTrain[:, P]
  xOrd = hcat(xOrdTrain, xTest)
  skeletons = construct_skeletons(colptr, rowval, distances, λ)
  L = assembleL(skeletons, xOrdTrain, covfunc!, σ)
  μOrd = μ
  ΣOrd = Σ
  yOrd = y[Block(1)][P]
  σOrd = σ[P]

  TMat = pairwise(Euclidean(), xOrdTrain, xTest, dims=2)
  covfunc!(TMat)
  CMat = pairwise(Euclidean(), xTest, dims=2)
  covfunc!(CMat)

  μNoPred = vec(yOrd' * L * L' * TMat)
  ΣNoPred = CMat - TMat' * L * L' * TMat

  # @show norm(μOrd - μPredFirst) / norm(μOrd)
  # @show norm(μOrd - μNoPred)
  # @show norm(ΣOrd - ΣNoPred)
  @show mean(abs.(μOrd - μNoPred))
  @show mean(abs.(ΣOrd - ΣNoPred))


# save("./out/jld/newname.jld", 
#      "ρList", ρList,
#      "meanErrorClosePredFirst", meanErrorClosePredFirst,
#      "medianErrorClosePredFirst", medianErrorClosePredFirst,
#      "meanErrorClosePredLast", meanErrorClosePredLast,
#      "medianErrorClosePredLast", medianErrorClosePredLast,
#      "meanErrorFarPredFirst", meanErrorFarPredFirst,
#      "medianErrorFarPredFirst", medianErrorFarPredFirst,
#      "meanErrorFarPredLast", meanErrorFarPredLast,
#      "medianErrorFarPredLast", medianErrorFarPredLast)
