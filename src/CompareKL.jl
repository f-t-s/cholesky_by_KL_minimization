include("SortSparse.jl")
include("KoLesky.jl")
include("Cholesky.jl")
include("Utils.jl")
include("CovFuncs.jl")
using LinearAlgebra
using RecursiveArrayTools
using Distributions
using Distances
using Random
using Match
using Profile
using JLD
using MAT 
using BlockArrays
Random.seed!(123)

NTrain = 10000
NTest = 1
N = NTest + NTrain
d =  2

xTrain = rand(d, NTrain)
xTestClose = rand(d, NTest)
xTestFar = 1.5 * ones(d,NTest) .+ 0.1 * rand(d, NTest)
xTest = xTestFar
x = hcat(xTrain, xTest)

l = 0.5
σ = 1e-16 * Diagonal(2.0 * ones(N))
maternOrder = "12"

covfunc12!(r) = matern12!(r,l)
covfunc32!(r) = matern32!(r,l)
covfunc52!(r) = matern52!(r,l)

# Make sure to avoid NANs
@match maternOrder begin
  "12" => (covfunc! = covfunc12!)
  "32" => (covfunc! = covfunc32!)
  "52" => (covfunc! = covfunc52!)
end



ρ = 10.0
λ = 1.5

# =====================================
# Screw that, try different dataset:
# =====================================
h = 1 / NTrain
xTrain = h : h : 1 
xTrain = vcat(sin.(2 * π * xTrain)', 
              2 * cos.(2 * π * xTrain)')
xTest = 0.1 * randn(2,NTest) 
x = hcat(xTrain, xTest)




# =====================================
# Prediction variables last
# =====================================


# reordering points and forming skeletons
colptr, rowval, P, revP, distances = sortSparseRev(xTrain, xTest, ρ, 1)
xOrd = x[:, P]
skeletons = construct_skeletons(colptr, rowval, distances, λ)

# Creating 
Θ = (pairwise(Euclidean(), xOrd, dims=2))
covfunc!(Θ)
Θ = Θ

L = assembleL(skeletons, xOrd, covfunc!)

ΘPredLast = inv(Matrix(L * L'))

@show KL(Θ, ΘPredLast)
@show KL(ΘPredLast, Θ)

# =====================================
# Computing posterior mean for prediction variables first
# =====================================

colptr, rowval, P, revP, distances = sortSparseRev(xTrain, ρ, 1)
xTrainOrd = xTrain[:, P]
xOrd = hcat(xTrainOrd, xTest)

L = SparseMatrixCSC(NTrain, NTrain, colptr, rowval, ones(size(rowval, 1)))
L = vcat(L, ones(NTest, NTrain))
L = tril(hcat(L, ones(N, NTest)))

colptr = L.colptr
rowval = L.rowval
distances = vcat(distances, fill(Inf, NTest))


skeletons = construct_skeletons(colptr, rowval, distances, λ)

# Creating 
Θ = (pairwise(Euclidean(), xOrd, dims=2))
covfunc!(Θ)
Θ = Θ

L = assembleL(skeletons, xOrd, covfunc!)

ΘPredFirst = inv(Matrix(L * L'))

@show KL(Θ, ΘPredFirst)
@show KL(ΘPredFirst, Θ)

# =====================================
# Don't include prediction variables at all
# =====================================

# reordering points and forming skeletons
colptr, rowval, P, revP, distances = sortSparseRev(xTrain, ρ, 1)
xOrdTrain = xTrain[:, P]
xOrd = hcat(xOrdTrain, xTest)
skeletons = construct_skeletons(colptr, rowval, distances, λ)
L = assembleL(skeletons, xOrdTrain, covfunc!, σ)

# Creating 
Θ = (pairwise(Euclidean(), xOrd, dims=2))
covfunc!(Θ)
Θ = Θ


TMat = pairwise(Euclidean(), xOrdTrain, xTest, dims=2)
covfunc!(TMat)
CMat = pairwise(Euclidean(), xTest, dims=2)
covfunc!(CMat)

ΘNoPred = inv(Matrix(L * L'))
ΘNoPred = vcat(ΘNoPred, TMat')
ΘNoPred = hcat(ΘNoPred, vcat(TMat, CMat))

@show KL(Θ, ΘNoPred)
@show KL(ΘNoPred, Θ)