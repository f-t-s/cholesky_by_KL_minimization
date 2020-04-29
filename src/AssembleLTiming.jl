include("SortSparse.jl")
include("KoLesky.jl")
include("Cholesky.jl")
include("CovFuncs.jl")
using LinearAlgebra
using RecursiveArrayTools
using Distributions
using Distances
using Random
using Profile
using ProfileView
using JLD
using MAT 
using IterativeSolvers
using BlockArrays
Random.seed!(123)

# Creating uniform random data
NTrain = 1000000
d =  2
NTest = 50
N = NTest + NTrain
x = rand(d, N)
setSplit = randperm(N)
trainInds = setSplit[1:NTrain]
testInds = setSplit[(NTrain + 1) : end]
xTrain = x[ :, trainInds ]
xTest = x[:, testInds ]
x = hcat(xTrain, xTest)
y = randn(N)
y = PseudoBlockArray(y, [NTest, NTrain])

ρ = 3.0
λ = 1.5


# reordering points and forming skeletons
# @profview colptr, rowval, P, revP, distances = sortSparseRev(xTrain, xTest, ρ, 1)
@time colptr, rowval, P, revP, distances = sortSparseRev(xTrain, xTest, ρ, 1)
x = x[:, P]

LTest = SparseMatrixCSC(N, N, colptr, rowval, ones(length(rowval)))
LTest22 = LTest[(NTest + 1) : end, (NTest + 1) : end]
 
xTest = x[:, 1:NTest]
xTrain = x[:, (NTest + 1) : end]
skeletons = construct_skeletons(colptr, rowval, distances, λ)

l = 0.1
# σ = vcat(zeros(NTest), fill(10.0, NTrain))
σ = vcat(zeros(NTest), 1.0 * exp.(randn(NTrain)) .+ 1e-4)
σ = PseudoBlockArray(σ, [NTest, NTrain])

function covfunc!(r)
  matern32!(r,l)
end

LinearAlgebra.BLAS.set_num_threads(1)
# @profview L = assembleL(skeletons, x, covfunc!)
@time L = assembleL(skeletons, x, covfunc!)
# @time LNaive = assembleL(skeletons, x, covfunc!, σ)
# @time L22 = L[(NTest + 1) : end, (NTest + 1) : end]
# @time A22 = L22 * L22' .+ spdiagm(0 => 1 ./ max.(σ[Block(2)], eps(eltype(σ))))
# # @time U22 = sparse(triu(A22))
# @time U22 = squareSparse(L22)+ spdiagm(0 => 1 ./ max.(σ[Block(2)], 1e-30))
# @time icholU_high_level!(U22)
# 
# 
# # The vecchia posterior mean
# z = (L22 * (L22' * y[Block(2)]))
# z = cg(A22 , z, Pl = IChol(U22',U22), verbose=true, maxiter=5, tol=1e-16)
# z = (Diagonal(σ[Block(2)]))\ z
# μVechia = (L' \ (L \ (vcat(zero(y[Block(1)]), z))))[1:NTest]
# # display(μVechia)
# @show mean(abs.(μVechia))
# @show median(abs.(μVechia))
# 
# 
# # The naive application of vecchia
# μNaive = -LNaive[1:NTest, 1:NTest]' \ (LNaive[(NTest + 1) : end, 1 : NTest]' * y[Block(2)])
# # display(μNaive)
# @show mean(abs.(μNaive))
# @show median(abs.(μNaive))
# 