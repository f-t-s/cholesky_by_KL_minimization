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
Random.seed!(123)

N = 10000
d = 2
l = 0.5
σ = exp.(randn(N))
ρList = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
λ = 1.5

# Creating uniform random data
x = rand(d, N)

# Creating output arrays
NNZ = zeros(length(ρList))
KLTrueNaive = zeros(3, length(ρList))
KLNaiveTrue = zeros(3, length(ρList))
KLTrueApp = zeros(3, length(ρList))
KLAppTrue = zeros(3, length(ρList))
KLTrueLarge = zeros(3, length(ρList))
KLLargeTrue = zeros(3, length(ρList))
KLTrueExact = zeros(3, length(ρList))
KLExactTrue = zeros(3, length(ρList))

σ .= max.(σ, eps(Float64))

covfunc12!(r) = matern12!(r,l)
covfunc32!(r) = matern32!(r,l)
covfunc52!(r) = matern52!(r,l)


for (iOrder, maternOrder) in enumerate(["12", "32", "52"])
  # Make sure to avoid NANs
  @match maternOrder begin
    "12" => (covfunc! = covfunc12!)
    "32" => (covfunc! = covfunc32!)
    "52" => (covfunc! = covfunc52!)
  end
  
  for (iρ, ρ) in enumerate(ρList)

    # reordering points and forming skeletons
    colptr, rowval, P, revP, distances = sortSparseRev(x, ρ, 1)
    xOrd = x[:, P]
    skeletons = construct_skeletons(colptr, rowval, distances, λ)
  
    # Creating reference solution 
    Θ = covfunc!(pairwise(Euclidean(), xOrd, dims=2)) + Diagonal(σ)
  
    # Creating naive vecchia approximation
    LNaive = assembleL(skeletons, xOrd, covfunc!, σ)
    ΘNaive = inv(Matrix(LNaive)') * inv(Matrix(LNaive))
  
    # Creating "smart" vecchia approximation
    L = assembleL(skeletons, xOrd, covfunc!)
    U = squareSparse(L) + spdiagm(0 => 1 ./ σ)
    icholU_high_level!(U)
    noisecov = NoiseCov(L, U, σ)
    ΘExact = exactMatrix(noisecov)
    ΘApp = appMatrix(noisecov)
  
    # Creating smart vecchia approximation with larger stencil for IC0
    ULarge = triu(L * L' + spdiagm(0 => 1 ./ σ))
    icholU_high_level!(ULarge)
    noisecovLarge = NoiseCov(L, ULarge, σ)
    ΘLarge = appMatrix(noisecovLarge)
  
    KLTrueNaive[iOrder, iρ] = KL(Θ, ΘNaive)
    KLNaiveTrue[iOrder, iρ] = KL(ΘNaive, Θ)

    KLTrueApp[iOrder, iρ] = KL(Θ, ΘApp)
    KLAppTrue[iOrder, iρ] = KL(ΘApp, Θ)
  
    KLTrueLarge[iOrder, iρ] = KL(Θ, ΘLarge)
    KLLargeTrue[iOrder, iρ] = KL(ΘLarge, Θ)

    KLTrueExact[iOrder, iρ] = KL(Θ, ΘExact)
    KLExactTrue[iOrder, iρ] = KL(ΘExact, Θ)
  end
end

save("./out/jld/KLVaryRho.jld", 
     "ρList", ρList,
     "KLTrueNaive", KLTrueNaive,
     "KLNaiveTrue", KLNaiveTrue,
     "KLTrueApp", KLTrueApp,
     "KLAppTrue", KLAppTrue,
     "KLTrueLarge", KLTrueLarge,
     "KLLargeTrue", KLLargeTrue,
     "KLTrueExact", KLTrueExact,
     "KLExactTrue", KLExactTrue)
