using JLD
using Random
import Distributions.Bernoulli


let 
  accuracy_exact = zeros(6)
  timing_exact = zeros(6)
  nnz_exact = zeros(6)

  accuracy_predict = zeros(6,3)
  timing_predict = zeros(6,3)
  nnz_predict = zeros(6,3)

  ########################################
  # Setting up the problem
  ########################################
  Random.seed!(123)
  include("BEMUtils.jl")
  λ = 1.0
  @show Threads.nthreads()

  n_test = 100
  test_points = points([1.0, 1.0, 1.0] .- [1.0, 0.1, 0.1] .* rand(3, n_test))

  n_charges = 1000
  charges_weights = [sign.(rand(Bernoulli(0.8), n_charges) .- 0.5);
                     sign.(rand(Bernoulli(0.3), n_charges) .- 0.5)]
  charges_points = points([(rand(3, n_charges) .* [1.0, 1.0, 0.1] .+ [0.0, 0.0, 1.0])';
                           (rand(3, n_charges) .* [1.0, 0.1, 1.0] .+ [0.0, 1.0, 0.0])']')

  u_true = assembleMatrix(test_points, charges_points) * charges_weights
  ########################################

  for q ∈ 3:8
    for ρ ∈ [1.0, 2.0, 3.0]
    hwv = haarCube(q)

    ########################################
    # Loading the "approximate approximation"
    ########################################

      println("Loading ordering from file")
      ld = load("./out/jld/BEM_sort_q_$(q).jld")
      P = ld["P"] 
      revP = ld["revP"]
      distances = ld["distances"]
      hwv = hwv[P]

      println("loading boundary data from file")
      ld = load("./out/jld/BEM_boundaryData_q_$(q).jld")
      boundaryData = ld["boundaryData"]


      println("loading prediction from file")
      ld = load("./out/jld/BEM_predict_q_$(q)_rho_$(Int(ρ * 10)).jld")
      u_predict = ld["u_predict"]
      time_predict = ld["time_predict"] 
      length_rowval = ld["length_rowval"]


      accuracy_predict[q - 2, Int(ρ)] = mean((u_true - u_predict).^2)
      timing_predict[q - 2, Int(ρ)] = time_predict
      nnz_predict[q - 2, Int(ρ)] = length_rowval
    ########################################

    ########################################
    # Loading the "exact approximation" if existant
    ########################################
    if isfile("./out/jld/BEM_exact_q_$(q).jld")
      ld = load("./out/jld/BEM_exact_q_$(q).jld")
      Θ = ld["Θ"]
      time_assemble_dense = ld["timing_assemble_dense"]

      time_compute_dense = @elapsed u_BEM_exact = assembleMatrix(test_points, hwv) * (Θ \ boundaryData)
      time_dense = time_compute_dense + time_assemble_dense

      @show q
      @show accuracy_exact[q - 2] = mean((u_true - u_BEM_exact).^2)
      timing_exact[q - 2] = time_compute_dense + time_dense
    else
      accuracy_exact[q - 2] = NaN
      timing_exact[q - 2] = NaN
    end
    nnz_exact[q - 2] = (length(hwv) + 1) * length(hwv) / 2
    ########################################
    end
  end
save("./out/jld/BEM_Plot.jld",
     "accuracy_exact", accuracy_exact,
     "timing_exact", timing_exact,
     "nnz_exact", nnz_exact,
     "accuracy_predict", accuracy_predict,
     "timing_predict", timing_predict,
     "nnz_predict", nnz_predict)
end
