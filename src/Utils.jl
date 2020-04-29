include("./Cholesky.jl")
using LinearAlgebra
using IntervalSets
using SpectralGaussianSimulation
using DirectGaussianSimulation

function exactMatrix(A::NoiseCov)
  return  Matrix(A.L * A.L') \ Matrix((A.L * A.L' + spdiagm(0 => A.invσ)) * spdiagm(0 => (1 ./ A.invσ)))
end

function appMatrix(A::NoiseCov)
  return  Matrix(A.L * A.L') \ Matrix(A.U' * A.U * spdiagm(0 => 1 ./ A.invσ))
end

function KL(M,N)
  N = Hermitian(N)
  M = Hermitian(M)
  try 
    return 1/2 * (tr(N \ M) + logdet(N) - logdet(M) - size(M, 1))
  catch err
    if isa(err, DomainError)
      return typemax(eltype(M))
    end
  end
end

function KL(M,N)
  N = Hermitian(N)
  M = Hermitian(M)
  try 
    return 1/2 * (tr(N \ M) + logdet(N) - logdet(M) - size(M, 1))
  catch err
    if isa(err, DomainError)
      return typemax(eltype(M))
    end
  end
end


# Samples from a given variogram using the FFT, on the unit square. Leaves a padding of length n on either side to decrease artifacts due to fft sampling
function specSampleGrid2D(n, NSamples, padFact::Int, var)
  x = zeros(2, n^2)
  Y = zeros(n^2, NSamples)
  for k in Iterators.partition(1:NSamples, 10)
    simProblem = SimulationProblem(RegularGrid((-padFact * 1.0, -padFact * 1.0), (1.0 + padFact * 1.0, 1.0 + padFact * 1.0), dims=((1 + 2 * padFact) * n, (1 + 2 * padFact) * n)), :z => Float64, length(k))

    simSolver = SpecGaussSim(:z => (variogram=var,))
    simSolution = solve(simProblem, simSolver)

    xRaw = coordinates(domain(simSolution))
    YRaw = reduce(hcat, simSolution.realizations[:z])

    xInds = xRaw .∈ [0.0 .. 1.0]
    xInds = findall(vec(xInds[1, :] .* xInds[2, :]))

    @assert  length(xInds) == n^2 

    # only for first iteration
    if k[1] == 1
      x .= xRaw[:, xInds]
      Y[:, k] .= YRaw[xInds, :]
    else
      @assert x == xRaw[:, xInds]
      Y[:, k] .= YRaw[xInds, :]
      GC.gc()
    end
  end
  return x, Y
end




