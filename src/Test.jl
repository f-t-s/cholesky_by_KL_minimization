include("SortSparse.jl")
include("KoLesky.jl")
using LinearAlgebra
using Distributions
using Distances
using Random
using Profile
using JLD
using MAT 
Random.seed!(123)

NTrain = 8000
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

ρ = 3.5
λ = 1.0

l = 0.1
cov = r -> exp(-r/l)

@time colptr, rowval, P, revP, distances = sortSparseRev(xTrain,ρ, 1)
xTrain = xTrain[:, P]
yTrain = yTrain[P]
x = hcat(xTrain, xTest)

skeletons = construct_skeletons(colptr, rowval, distances, λ)
colptr = 0
length(skeletons)
rowval = 0
GC.gc()

#Comparison to full kodel
LinearAlgebra.BLAS.set_num_threads(8)
@time begin
  Θ = cov.(pairwise(Euclidean(), x, dims=2))
  μFull = Θ[(NTrain + 1) : end, 1 : NTrain] * ( Θ[1 : NTrain, 1 : NTrain] \ yTrain) 
end

Θ = 0
GC.gc()
@show mean(abs.(yTest - μFull))
@show mean(abs.(yTest - μFull)) / std(yTest)

@time μ, σ = predict(skeletons, (NTrain+1): N, yTrain, x, cov)

mean(abs.(yTest - μ)) / std(yTest)
mean(abs.(yTest - μ))