include("SortSparse.jl")
include("KoLesky.jl")
include("CovFuncs.jl")
using LinearAlgebra
using Distributions
using Distances
using Random
using Profile
using JLD
using MAT 
using Match
Random.seed!(123)

NTrain = 2000
NTest = 100
N = NTest + NTrain
d =  2
y = randn(N)
x = randn(d, N)

setSplit = randperm(N)
trainInds = setSplit[1:NTrain]
testInds = setSplit[(NTrain + 1) : end]

xTrain = x[ :, trainInds ]
yTrain = y[ trainInds ]
xTest = x[:, testInds ]
yTest = y[ testInds ]

ρ = 2.5
λ = 1.5

l = 0.5


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


@time colptr, rowval, P, revP, distances = sortSparseRev(xTrain,ρ, 1)
xTrain = xTrain[:, P]
yTrain = yTrain[P]
x = hcat(xTrain, xTest)

skeletons = construct_skeletons(colptr, rowval, distances, λ)
# TODO: Remove again
# skeletons = singleParents(skeletons)

colptr = 0
length(skeletons)
rowval = 0
GC.gc()

#Comparison to full model
LinearAlgebra.BLAS.set_num_threads(8)
@time begin
  Θ = covfunc!(pairwise(Euclidean(), x, dims=2))
  μFull = Θ[(NTrain + 1) : end, 1 : NTrain] * ( Θ[1 : NTrain, 1 : NTrain] \ yTrain) 
end

Θ = 0
GC.gc()
# @show mean(abs.(yTest - μFull))
# @show mean(abs.(yTest - μFull)) / std(yTest)

@time μ, σ = predict(skeletons, (NTrain+1): N, yTrain, x, covfunc!)

# @show mean(abs.(yTest - μ)) / std(yTest)
# @show mean(abs.(yTest - μ))

@show norm(μFull - μ) / norm(μFull)
@show norm(μFull - μ) 