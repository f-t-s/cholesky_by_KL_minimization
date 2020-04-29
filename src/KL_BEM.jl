import Distributions.Bernoulli
using JLD
using ArgParse
using Random

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "q"
            help = "number of levels"
            arg_type = Int
            required = true
        "ρ"
            help = "oversampling radius"
            arg_type = Float64
            required = true
        "dense"
            help = "compute reference solution using dense linear algebra"
            arg_type = Bool
            required = true
    end

    return parse_args(s)
end

parsed_args = parse_commandline()
println("Parsed args:")
for (arg,val) in parsed_args
    println("  $arg  =>  $val")
end

for key in keys(parsed_args)
  symb = Symbol(key)
  data = parsed_args[key]
  @eval $(symb) = $(data)
end


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






  #Either loads The exact solution, or computes it if it is not available
  if isfile("./out/jld/BEM_predict_q_$(q)_rho_$(Int(ρ * 10)).jld")
    # Loads the ordering 
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

  else 
    println("Computing ordering from scratch")
    colptr, rowval, P, revP, distances = sortSparseRev(hwv, ρ)
    save("./out/jld/BEM_sort_q_$(q).jld", 
    "P", P,
    "revP", revP,
    "distances", distances) 


    # Change distance to ensure that no grouping into skeletons takes place.

    distances .+= 1e-12 * (1 : length(distances))
    length_rowval = length(rowval)
    skeletons = construct_skeletons(colptr, rowval, distances, λ)
    hwv = hwv[P]

    println("computing boundary data from scratch")
    boundaryData = assembleMatrix(hwv, charges_points) * charges_weights
    save("./out/jld/BEM_boundaryData_q_$(q).jld", "boundaryData", boundaryData)


    println("computing prediction from scratch")
    time_predict = @elapsed u_predict = predict(skeletons, boundaryData, hwv, test_points)[1]
    save("./out/jld/BEM_predict_q_$(q)_rho_$(Int(ρ * 10)).jld", "u_predict", u_predict, "time_predict", time_predict, "length_rowval", length_rowval) 
  end



  u_true = assembleMatrix(test_points, charges_points) * charges_weights

  # >>>>>>>>>>>>>>>>>>>>>>>> dense part >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
  #Either loads The exact solution, or computes it if it is not available
  if dense
    if isfile("./out/jld/BEM_exact_q_$(q).jld")
      println("Loading full matrix from file")
      ld = load("./out/jld/BEM_exact_q_$(q).jld")
      Θ = ld["Θ"]
      time_assemble_dense = ld["timing_assemble_dense"]
    else
      println("Computing full matrix from scratch")
      time_assemble_dense = @elapsed Θ = assembleMatrix(hwv)
      save("./out/jld/BEM_exact_q_$(q).jld", "Θ", Θ, "timing_assemble_dense", time_assemble_dense)
    end

    time_compute_dense = @elapsed u_BEM_exact = assembleMatrix(test_points, hwv) * (Θ \ boundaryData)
    time_dense = time_compute_dense + time_assemble_dense
    @show mean((u_true - u_BEM_exact).^2)
    @show mean((u_predict - u_BEM_exact).^2)
    @show time_dense
  end
  # <<<<<<<<<<<<<<<<<<<<<<<< dense part <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

  @show mean((u_true - u_predict).^2)
  @show time_predict

  @show 2 * length_rowval / length(hwv) / length(hwv)
end 


