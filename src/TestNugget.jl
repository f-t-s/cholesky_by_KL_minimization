include("SortSparse.jl")
include("KoLesky.jl")
include("Cholesky.jl")
using LinearAlgebra
using RecursiveArrayTools
using Distributions
using Distances
using Random
using Profile
using JLD
using MAT 
using BlockArrays
Random.seed!(123)

NTrain = 10000
NTest = 5
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

n = 10
h = 1 / (n - 1)

xTrain = Matrix(VectorOfArray([[i * h; j * h] for i in 1:n, j in 1:n][:]))
NTrain = size(xTrain,2)
yTrain = randn(NTrain)
N = NTrain + NTest

l = 1.0
σ = 1e-15
cov = r -> exp(-r/l) # + Float64(r <= 1e-10)

rhovals = [2.0, 3.0, 4.0, 5.0]

ρ = 3.0
λ = 1.3


# reordering points and forming skeletons
@time colptr, rowval, P, revP, distances = sortSparseRev(xTrain,ρ, 1)
xTrain = xTrain[:, P]
yTrain = yTrain[P]
x = hcat(xTest, xTrain)
skeletons = construct_skeletons(colptr, rowval, distances, λ)

# Creating 
Θ = cov.(pairwise(Euclidean(), x, dims=2))
L_Θ = cholesky(Matrix(Θ))
y = L_Θ.L * randn(N)
Θ = Θ + σ * I

Θ = PseudoBlockArray(Θ, [NTest, NTrain], [NTest, NTrain])
y = PseudoBlockArray(y, [NTest, NTrain])
y[Block(2)] += √(σ) * randn(NTrain)

L = assembleL(skeletons, xTrain, cov)
U = sparse(triu(L * L') + I / max(σ, 1e-30))
icholU_high_level!(U)

μTrue = Θ[Block(1,2)] * (Θ[Block(2,2)] \ y[Block(2)]) 


# Direct application of Vecchia
# μ = Θ[Block(1,2)] * (L * (L' * y[Block(2)]))

# Version with Cholesky factor
μ = Θ[Block(1,2)] * (σ \ (  U \ (U' \ (L * (L' * y[Block(2)])))))

display(μTrue)
display(μ)
display(y[Block(1)])

@show norm(μTrue - μ) / norm(μTrue)

