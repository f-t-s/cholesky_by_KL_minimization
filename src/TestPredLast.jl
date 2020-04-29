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
using JLD
using MAT 
using IterativeSolvers
using BlockArrays
Random.seed!(123)

# Creating uniform random data
NTrain = 2000
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

# # Creating gridded data 
# n = 20
# h = 1 / (n - 1)
# xTrain = Matrix(VectorOfArray([[i * h; j * h] for i in 1:n, j in 1:n][:]))
# NTrain = size(xTrain,2)
# NTest = 50
# xTest = rand(2, NTest)
# N = NTrain + NTest
# x = hcat(xTrain, xTest) # .+ 0.5 * (1/sqrt(N)) * randn(2, N)


ρ = 3.0
λ = 1.5


# reordering points and forming skeletons
@time colptr, rowval, P, revP, distances = sortSparseRev(xTrain, xTest, ρ, 1)
x = x[:, P]

LTest = SparseMatrixCSC(N, N, colptr, rowval, ones(length(rowval)))
LTest22 = LTest[(NTest + 1) : end, (NTest + 1) : end]

xTest = x[:, 1:NTest]
xTrain = x[:, (NTest + 1) : end]
skeletons = construct_skeletons(colptr, rowval, distances, λ)

l = 1.1
σ = vcat(zeros(NTest), fill(1000.0, NTrain))
# σ = vcat(zeros(NTest), max.(0.1 * exp.(randn(NTrain)), eps(Float64)))
σ = PseudoBlockArray(σ, [NTest, NTrain])
# covfunc = r -> exp(-r/l) 
# covfunc = r -> matern12!(r,l) 
function covfunc!(r)
  matern12!(r,l)
end
 
# Creating 
Θ = covfunc!(pairwise(Euclidean(), x, dims=2))
L_Θ = cholesky(Matrix(Θ))
y = L_Θ.L * randn(N)
Θ = Θ + Diagonal(σ)
 
# Creating data
Θ = PseudoBlockArray(Θ, [NTest, NTrain], [NTest, NTrain])
y = PseudoBlockArray(y, [NTest, NTrain])
y[Block(2)] .+= Diagonal(sqrt.(σ[Block(2)])) * randn(NTrain)
 
L = assembleL(skeletons, x, covfunc!)
LNaive = assembleL(skeletons, x, covfunc!, σ)
L22 = L[(NTest + 1) : end, (NTest + 1) : end]
A22 = L22 * L22' + Diagonal(1 ./ σ[Block(2)])
# U22 = sparse(triu(A22))
U22 = squareSparse(L22)+ Diagonal(1 ./ σ[Block(2)])

icholU_high_level!(U22)
# U22 = UpperTriangular(U22)
U22Exct = cholesky(Matrix(A22)).U


# The exact posterior mean 
μFull = Θ[Block(1,2)] * (Θ[Block(2,2)] \ y[Block(2)]) 
# display(μFull)

# The vecchia posterior mean
# z = (Diagonal(σ[Block(2)]) \ (  U22 \ (U22' \ (L22 * (L22' * y[Block(2)])))))
noisecov = NoiseCov(L22, U22, σ[Block(2)])
# z = (L22 * (L22' * y[Block(2)]))
# z = cg(A22 , z, Pl = IChol(U22',U22), verbose=true, maxiter=10, tol=1e-16)
# z = (Diagonal(σ[Block(2)]))\ z

z = noisecov \ (y[Block(2)])

μVechia = (L' \ (L \ (vcat(zero(y[Block(1)]), z))))[1:NTest]
# display(μVechia)
@show mean(abs.(μFull - μVechia))
@show median(abs.(μFull - μVechia))

# The vecchia posterior mean
# z = (Diagonal(σ[Block(2)]) \ ( U22Exct \ (U22Exct' \ (L22 * (L22' * y[Block(2)])))))
z = (L22 * (L22' * y[Block(2)]))
z = cg(A22 , z, Pl = IChol(U22Exct',U22Exct), verbose=true, maxiter=10)
z = (Diagonal(σ[Block(2)]))\ z

μVechiaExct = (L' \ (L \ (vcat(zero(y[Block(1)]), z))))[1:NTest]
# display(μVechiaExct)
@show mean(abs.(μFull - μVechiaExct))
@show median(abs.(μFull - μVechiaExct))

# The naive application of vecchia
μNaive = -LNaive[1:NTest, 1:NTest]' \ (LNaive[(NTest + 1) : end, 1 : NTest]' * y[Block(2)])
# display(μNaive)
@show mean(abs.(μFull - μNaive))
@show median(abs.(μFull - μNaive))

# # Direct application of Vecchia
# # μ = Θ[Block(1,2)] * (L * (L' * y[Block(2)]))
# 
# # Version with Cholesky factor
# μ = Θ[Block(1,2)] * (σ \ (  U \ (U' \ (L * (L' * y[Block(2)])))))
# 
# display(μTrue)
# display(μ)
# display(y[Block(1)])
# 
# @show norm(μTrue - μ) / norm(μTrue)
# 
# 