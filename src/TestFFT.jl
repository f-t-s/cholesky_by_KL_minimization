# Testing whether SpectralGaussianSimulation does what I think it does
using GeoStats
using SpectralGaussianSimulation
using DirectGaussianSimulation
using Distances
include("CovFuncs.jl")
include("Utils.jl")

n = 100
N = n^2
NSamples = 2000

l = 0.5
# covfunc!(r) = exponential!(r, l)
covfunc!(r) = matern32!(r, l)


# simProblem = SimulationProblem(RegularGrid((-2.0, -2.0.0, 0.0), (1.0, 1.0), dims=(n, n)), :z => Float64, NSamples)

# simSolver = SpecGaussSim(:z => (variogram=MaternVariogram(range=l, order=1.5),))

# simSolution = solve(simProblem, simSolver)
# x = coordinates(domain(simSolution))
# Y = reduce(hcat, simSolution.realizations[:z])

x, Y = specSampleGrid2D(n, NSamples, 3, MaternVariogram(range=l, order=1.5))

Θ = covfunc!(Distances.pairwise(Euclidean(), x[:, inds], dims=2))
YExact = rand(MvNormal(Θ), NSamples)


ΘEmp = cov(YExact, dims=2)
ΘEmpApp = cov(Y[inds, :], dims=2)

@show norm(Θ - ΘEmp) / norm(Θ)
@show norm(Θ - ΘEmpApp) / norm(Θ)