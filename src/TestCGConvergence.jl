include("SortSparse.jl")
include("KoLesky.jl")
include("Cholesky.jl")
include("CovFuncs.jl")
using LinearAlgebra
using Distributions
using Distances
using Random
using JLD 
using IterativeSolvers
using BlockArrays
using Match
Random.seed!(123)

# Creating uniform random data
N = 10000
NSamples = 1
max2iter = 1:10
d =  2
x = rand(d, N)
λ = 1.5
l = 0.5
σList = [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
ρList = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
covfunc12!(r) = matern12!(r,l)
covfunc32!(r) = matern32!(r,l)
covfunc52!(r) = matern52!(r,l)


maxError = zeros(10)
σBase = exp.(randn(N))
σBase .= max.(σBase, eps(Float64))
for maternOrder in enumerate(["12", "32", "52"])
  global maxError
  @match maternOrder begin
    "12" => (covfunc! = covfunc12!)
    "32" => (covfunc! = covfunc32!)
    "52" => (covfunc! = covfunc52!)
  end
  for ρ in ρList
    for σ in σList
      σ = σ * σBase
      for maxiter in max2iter
        # reordering points and forming skeletons
        colptr, rowval, P, revP, distances = sortSparseRev(x, ρ, 1)
        xOrd = x[:, P]
        skeletons = construct_skeletons(colptr, rowval, distances, λ)

        function covfunc!(r)
          matern12!(r,l)
        end

        # Creating Data
        Θ = covfunc!(pairwise(Euclidean(), xOrd, dims=2)) + Diagonal(σ)
        L_Θ = cholesky(Matrix(Θ))
        Y = L_Θ.L * randn(N, NSamples) 

        L = assembleL(skeletons, x, covfunc!)
        A = L * L' + Diagonal(1 ./ σ)
        U = squareSparse(L) + spdiagm(0 => (1 ./ σ))

        icholU_high_level!(U)
        UExct = cholesky(Matrix(A)).U

        for kSample = 1 : NSamples
          # The Exact vecchia posterior mean
          z = (L * (L' * Y[:, kSample]))
          ldiv!(IChol(UExct', UExct), z)
          z = σ .\ z
          μVechiaExct = (L' \ (L \ (z)))

          # The vecchia posterior mean after maxiter iterations of cg
          z = (L * (L' * Y))
          z = cg(A , z, Pl = IChol(U',U), verbose=false, maxiter=maxiter, tol=1e-99)
          z = σ .\ z
          μVechia = (L' \ (L \ (z)))
          maxError[maxiter] = max(maxError[maxiter], norm(μVechiaExct - μVechia) / norm(μVechiaExct))
        end
      end
    end
  end
end

save("./out/jld/TestCGConvergence.jld", 
     "σList", σList,
     "ρList", ρList,
     "maxError", maxError)