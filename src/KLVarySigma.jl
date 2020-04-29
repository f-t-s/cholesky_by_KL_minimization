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
σList = [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
ρ = 3.0
λ = 1.5

# Creating uniform random data
x = rand(d, N)

# reordering points and forming skeletons
colptr, rowval, P, revP, distances = sortSparseRev(x, ρ, 1)
xOrd = x[:, P]
skeletons = construct_skeletons(colptr, rowval, distances, λ)

KLTrueNaive = zeros(3, length(σList))
KLNaiveTrue = zeros(3, length(σList))
KLTrueApp = zeros(3, length(σList))
KLAppTrue = zeros(3, length(σList))
KLTrueLarge = zeros(3, length(σList))
KLLargeTrue = zeros(3, length(σList))
KLTrueExact = zeros(3, length(σList))
KLExactTrue = zeros(3, length(σList))


σBase = exp.(randn(N))
σBase .= max.(σBase, eps(Float64))

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
  
  for (iScale, σScale) in enumerate(σList)
    σ = σScale * σBase
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
  
    KLTrueNaive[iOrder, iScale] = KL(Θ, ΘNaive)
    KLNaiveTrue[iOrder, iScale] = KL(ΘNaive, Θ)

    KLTrueApp[iOrder, iScale] = KL(Θ, ΘApp)
    KLAppTrue[iOrder, iScale] = KL(ΘApp, Θ)
  
    KLTrueLarge[iOrder, iScale] = KL(Θ, ΘLarge)
    KLLargeTrue[iOrder, iScale] = KL(ΘLarge, Θ)

    KLTrueExact[iOrder, iScale] = KL(Θ, ΘExact)
    KLExactTrue[iOrder, iScale] = KL(ΘExact, Θ)
  end
end

save("./out/jld/KLVarySigma.jld", 
     "σList", σList,
     "KLTrueNaive", KLTrueNaive,
     "KLNaiveTrue", KLNaiveTrue,
     "KLTrueApp", KLTrueApp,
     "KLAppTrue", KLAppTrue,
     "KLTrueLarge", KLTrueLarge,
     "KLLargeTrue", KLLargeTrue,
     "KLTrueExact", KLTrueExact,
     "KLExactTrue", KLExactTrue)
