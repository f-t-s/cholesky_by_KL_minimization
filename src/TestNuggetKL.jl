include("SortSparse.jl")
include("KoLesky.jl")
include("Cholesky.jl")
include("Utils.jl")
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

NTrain = 2000
NTest = 5
N = NTest + NTrain
d =  2
x = randn(d, N)

l = 1.0
σ = Diagonal(2.0 * ones(N))
cov = r -> exp(-r/l)


ρ = 4.0
λ = 1.3


# reordering points and forming skeletons
@time colptr, rowval, P, revP, distances = sortSparseRev(x, ρ, 1)
x = x[:, P]
skeletons = construct_skeletons(colptr, rowval, distances, λ)

# Creating 
Θ = cov.(pairwise(Euclidean(), x, dims=2))
Θ = Θ + σ

L = assembleL(skeletons, x, cov)
U = sparse(triu(L * L') + min.(inv(σ), 1e30))
icholU_high_level!(U)

ΘApprox = inv(Matrix(L' * L)) * U' * U * σ 

@show KL(Θ, ΘApprox)
@show KL(ΘApprox, Θ)


