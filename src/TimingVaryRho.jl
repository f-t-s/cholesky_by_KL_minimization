using JLD
using LinearAlgebra

# Setting number of threads to one
LinearAlgebra.BLAS.set_num_threads(1)

include("./SortSparse.jl")
include("./KoLesky.jl")
include("./CovFuncs.jl")

N = 1000000
x = rand(2,N)

ρList = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
timings = zeros(size(ρList))
nonz = zeros(size(ρList))
timingsAgg = zeros(size(ρList))
nonzAgg = zeros(size(ρList))
timingsBessel = zeros(size(ρList))
nonzBessel = zeros(size(ρList))
timingsBesselAgg = zeros(size(ρList))
nonzBesselAgg = zeros(size(ρList))

# Timings
for (ρInd, ρ) in enumerate(ρList)
  global timings
  global nonz

  covfunc!(r) = matern32!(r, 0.1)
  colptr, rowval, P, revP, distances =  sortSparseRev(x, ρ, 1)
  xLoc = x[:, P]
  skeletons = construct_skeletons(colptr, rowval, distances, 1.0)
  # Making sure everything is compiled
  if ρInd == 1
    @elapsed assembleL(skeletons, xLoc, covfunc!)
  end
  timings[ρInd] = @elapsed L = assembleL(skeletons, xLoc, covfunc!)
  nonz[ρInd] = nnz(L)
end

# Timings with aggregation
for (ρInd, ρ) in enumerate(ρList)
  global timingsAgg
  global nonzAgg
  covfunc!(r) = matern32!(r, 0.1)
  colptr, rowval, P, revP, distances =  sortSparseRev(x, ρ, 1)
  xLoc = x[:, P]
  skeletons = construct_skeletons(colptr, rowval, distances, 1.5)
  # Making sure everything is compiled
  if ρInd == 1
    @elapsed assembleL(skeletons, xLoc, covfunc!)
  end
  timingsAgg[ρInd] = @elapsed L = assembleL(skeletons, xLoc, covfunc!)
  nonzAgg[ρInd] = nnz(L)
end

# Timings with Bessel function
for (ρInd, ρ) in enumerate(ρList)
  global timingsBessel
  global nonzBessel
  function covfunc!(r)
    r .= matern.(r, 0.1, 1.5)
  end
  colptr, rowval, P, revP, distances =  sortSparseRev(x, ρ, 1)
  xLoc = x[:, P]
  skeletons = construct_skeletons(colptr, rowval, distances, 1.0)
  # Making sure everything is compiled
  if ρInd == 1
    @elapsed assembleL(skeletons, xLoc, covfunc!)
  end
  timingsBessel[ρInd] = @elapsed L = assembleL(skeletons, xLoc, covfunc!)
  nonzBessel[ρInd] = nnz(L)
end

# Timings with Bessel function and aggregation
for (ρInd, ρ) in enumerate(ρList)
  global timingsBesselAgg
  global nonzBesselAgg
  function covfunc!(r)
    r .= matern.(r, 0.1, 1.5)
  end
  colptr, rowval, P, revP, distances =  sortSparseRev(x, ρ, 1)
  xLoc = x[:, P]
  skeletons = construct_skeletons(colptr, rowval, distances, 1.5)
  # Making sure everything is compiled
  if ρInd == 1
    @elapsed assembleL(skeletons, xLoc, covfunc!)
  end
  timingsBesselAgg[ρInd] = @elapsed L = assembleL(skeletons, xLoc, covfunc!)
  nonzBesselAgg[ρInd] = nnz(L)
end

save("./out/jld/TimingVaryRho.jld",
     "ρList", ρList, 
     "timings", timings, 
     "nonz", nonz, 
     "timingsAgg", timingsAgg,
     "nonzAgg", nonzAgg,
     "timingsBessel", timingsBessel,
     "nonzBessel", nonzBessel,
     "timingsBesselAgg", timingsBesselAgg,
     "nonzBesselAgg", nonzBesselAgg)
