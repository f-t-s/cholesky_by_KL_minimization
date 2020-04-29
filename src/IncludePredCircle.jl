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
using ArgParse
Random.seed!(123)

s = ArgParseSettings()
@add_arg_table s begin
  "maternOrder"
    help = "The order of the matern kernel to be used. Should be in (\"12\", \"32\", \"52\")"
    required = true
end
maternOrder = parse_args(s)["maternOrder"]


NTrain = 10000
NTest = 1
NSamples = 100
N = NTrain + NTest
d = 2
l = 0.5
# For now, only properly implemented for $σ = 0$.
σ = 0.0 * exp.(randn(N))
λ = 1.5
ρ = 3.0

# Colums 
meanErrorPredFirst = zeros(NTest, NSamples)
stdErrorPredFirst = zeros(NTest, NSamples)
KLTrueApproxPredFirst = zeros(NSamples)
KLApproxTruePredFirst = zeros(NSamples)
meanErrorPredLast = zeros(NTest, NSamples)
stdErrorPredLast = zeros(NTest, NSamples)
KLTrueApproxPredLast = zeros(NSamples)
KLApproxTruePredLast = zeros(NSamples)
meanErrorNoPred = zeros(NTest, NSamples)
stdErrorNoPred = zeros(NTest, NSamples)
KLTrueApproxNoPred = zeros(NSamples)
KLApproxTrueNoPred = zeros(NSamples)

# Creating data on ellipsoid, testing data in the center
h = 1 / NTrain
xTrain = h : h : 1 
xTrain = vcat(sin.(2 * π * xTrain)', 
              2 * cos.(2 * π * xTrain)')
xTest = 0.1 * randn(2,NTest) 
x = hcat(xTrain, xTest)

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

for k = 1 : NSamples
  # =====================================
  # Starting with interpolation:
  # =====================================

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
  ΘOrd = Θ[P, P]
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

  ΘPredLast = inv(Matrix(L * L'))

  meanErrorPredLast[:, k] = abs.(μOrd - μPredLast)
  stdErrorPredLast[:, k] = abs.(diag(ΣOrd) - diag(ΣPredLast))
  KLTrueApproxPredLast[k] = KL(ΘOrd, ΘPredLast)
  KLApproxTruePredLast[k] = KL(ΘPredLast, ΘOrd)

  # =====================================
  # Computing posterior mean for prediction variables first
  # =====================================

  # reordering points and forming skeletons
  colptr, rowval, P, revP, distances = sortSparseRev(xTrain, ρ, 1)
  xOrdTrain = xTrain[:, P]
  xOrd = hcat(xOrdTrain, xTest)
  skeletons = construct_skeletons(colptr, rowval, distances, λ)
  μOrd = μ
  ΘOrd = Θ[vcat(P, (NTrain + 1) : N), vcat(P, (NTrain + 1) : N)]
  ΣOrd = Σ
  yOrd = y[Block(1)][P]
  σOrd = σ[P]

  μPredFirst, ΣPredFirst = predict(skeletons, (NTrain+1): N, yOrd, xOrd, covfunc!, σ)

  # Assembling the matrix implicit in predict, in order to compute KL divergence
  L = SparseMatrixCSC(NTrain, NTrain, colptr, rowval, ones(size(rowval, 1)))
  L = vcat(L, ones(NTest, NTrain))
  L = tril(hcat(L, ones(N, NTest)))
  colptr = L.colptr
  rowval = L.rowval
  distances = vcat(distances, fill(Inf, NTest))
  skeletons = construct_skeletons(colptr, rowval, distances, λ)
  L = assembleL(skeletons, xOrd, covfunc!)
  ΘPredFirst = inv(Matrix(L * L'))

  meanErrorPredFirst[:, k] = abs.(μOrd - μPredFirst)
  stdErrorPredFirst[:, k] = abs.(diag(ΣOrd) - ΣPredFirst)
  KLTrueApproxPredFirst[k] = KL(ΘOrd, ΘPredFirst)
  KLApproxTruePredFirst[k] = KL(ΘPredFirst, ΘOrd)

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
  ΘOrd = Θ[vcat(P, (NTrain + 1) : N), vcat(P, (NTrain + 1) : N)]
  ΣOrd = Σ
  yOrd = y[Block(1)][P]
  σOrd = σ[P]

  TMat = pairwise(Euclidean(), xOrdTrain, xTest, dims=2)
  covfunc!(TMat)
  CMat = pairwise(Euclidean(), xTest, dims=2)
  covfunc!(CMat)

  μNoPred = vec(yOrd' * L * L' * TMat)
  ΣNoPred = CMat - TMat' * L * L' * TMat

  ΘNoPred = inv(Matrix(L * L'))
  ΘNoPred = vcat(ΘNoPred, TMat')
  ΘNoPred = hcat(ΘNoPred, vcat(TMat, CMat))

  meanErrorNoPred[:, k] = abs.(μOrd - μNoPred)
  stdErrorNoPred[:, k] = abs.(ΣOrd - ΣNoPred)
  KLTrueApproxNoPred[k] = KL(ΘOrd, ΘNoPred)
  KLApproxTrueNoPred[k] = KL(ΘNoPred, ΘOrd)
end 

save("./out/jld/IncludePredCircle$maternOrder.jld", 
     "meanErrorNoPred", meanErrorNoPred,
     "stdErrorNoPred", stdErrorNoPred,
     "KLTrueApproxNoPred", KLTrueApproxNoPred,
     "KLApproxTrueNoPred", KLApproxTrueNoPred,
     "meanErrorPredFirst", meanErrorPredFirst,
     "stdErrorPredFirst", stdErrorPredFirst,
     "KLTrueApproxPredFirst", KLTrueApproxPredFirst,
     "KLApproxTruePredFirst", KLApproxTruePredFirst,
     "meanErrorPredLast", meanErrorPredLast,
     "stdErrorPredLast", stdErrorPredLast,
     "KLTrueApproxPredLast", KLTrueApproxPredLast,
     "KLApproxTruePredLast", KLApproxTruePredLast)