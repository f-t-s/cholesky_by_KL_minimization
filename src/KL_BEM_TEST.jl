import Distributions.Bernoulli
using JLD
using ArgParse
using Random

@show q = 6
@show ρ = 2.0

let 
  Random.seed!(123)

  include("BEMUtils.jl")

  λ = 1.0

  @show Threads.nthreads()

  hwv = haarCube(q)

  n_test = 100
  test_points = points([1.0, 1.0, 1.0] .- [1.0, 0.1, 0.1] .* rand(3, n_test))

  n_charges = 1000
  charges_weights = [sign.(rand(Bernoulli(0.8), n_charges) .- 0.5);
                     sign.(rand(Bernoulli(0.3), n_charges) .- 0.5)]
  charges_points = points([(rand(3, n_charges) .* [1.0, 1.0, 0.1] .+ [0.0, 0.0, 1.0])';
                           (rand(3, n_charges) .* [1.0, 0.1, 1.0] .+ [0.0, 1.0, 0.0])']')

  # Loads the ordering 
  println("Loading ordering from file")
  ld = load("./out/jld/BEM_sort_q_$(q).jld")
  P = ld["P"] 
  revP = ld["revP"]
  hwv = hwv[P]


  println("loading boundary data from file")
  ld = load("./out/jld/BEM_boundaryData_q_$(q).jld")
  boundaryData = ld["boundaryData"]

  colptr, rowval, P, revP, distances = sortSparseRev(hwv, ρ)

  # Change distance to ensure that no grouping into skeletons takes place.

  distances .+= 1e-12 * (1 : length(distances))
  length_rowval = length(rowval)
  skeletons = construct_skeletons(colptr, rowval, distances, λ)
  hwv = hwv[P]

  u_true = assembleMatrix(test_points, charges_points) * charges_weights

  # >>>>>>>>>>>>>>>>> Loading dense matrix and computing ref sol >>>>>>>>
  println("Loading full matrix from file")
  ld = load("./out/jld/BEM_exact_q_$(q).jld")
  Θ = ld["Θ"]
  time_assemble_dense = ld["timing_assemble_dense"]
  time_compute_dense = @elapsed u_BEM_exact = assembleMatrix(test_points, hwv) * (Θ \ boundaryData)
  time_dense = time_compute_dense + time_assemble_dense
  # <<<<<<<<<<<<<<<<< Loading dense matrix and computing the  <<<<<<<<


  # >>>>>>>>>>>>>>>>> computing prediction >>>>>>>>
  println("computing prediction from scratch")
  # time_predict = @elapsed u_predict = predict(skeletons, boundaryData, hwv, test_points, Θ)[1]
  time_predict = @elapsed u_predict = predict(skeletons, boundaryData, hwv, test_points, Θ)[1]
  # <<<<<<<<<<<<<<<<< computing prediction  <<<<<<<<

  # >>>>>>>>>>>>>>>>> computing prediction by approximating L >>>>>>>>
  println("computing prediction from scratch, by approximating L")
  @time L= assembleL(skeletons, hwv, Θ)
  u_approx_L = assembleMatrix(test_points, hwv) * (L * (L' * boundaryData))
  # <<<<<<<<<<<<<<<<< computing prediction by approximating L <<<<<<<<

  @show mean((u_predict - u_BEM_exact).^2)
  @show mean((u_approx_L - u_BEM_exact).^2)
  @show time_dense

  @show mean((u_true - u_BEM_exact).^2)
  @show mean((u_true - u_predict).^2)
  @show mean((u_true - u_approx_L).^2)
  @show time_predict

  @show 2 * length_rowval / length(hwv) / length(hwv)

  @show maximum([length(s.parents) for s in skeletons])
end 
