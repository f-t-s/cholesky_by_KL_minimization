import Distributions.Bernoulli
using JLD
using ArgParse
using Random
using Plots

Plots.default(show=false)

@show q = 6
@show ρ = 5.0

let 

  Random.seed!(123)

  include("BEMUtils.jl")
  distance = dist_sphere

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

  # TODO: Remove
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

  Θ_inv = inv(Θ)
  distance_matrix = zero(Θ)

  # @time for i = 1 : size(Θ, 1)
  @time  Threads.@threads for i = 1 : size(Θ, 1)
    for j = 1 : i
      distance_matrix[i, j] = distance(hwv[i], hwv[j]) / min(scale(hwv[i]), scale(hwv[j]))
    end
  end

  # inds = 1 : length(vec(distance_matrix))
   
  distance_matrix = Hermitian(distance_matrix, :L)
  Θ_inv = sqrt.(Diagonal(Θ_inv)) \ Θ_inv / sqrt.(Diagonal(Θ_inv))
  Θ = sqrt.(Diagonal(Θ)) \ Θ / sqrt.(Diagonal(Θ))

  plane_indices = [i for i in 1 : length(hwv) if hwv[i].v[3] ≈ 1.0]
  Θ_inv_plane = Θ_inv[plane_indices, plane_indices]
  Θ_plane = Θ[plane_indices, plane_indices]
  distance_matrix_plane = distance_matrix[plane_indices, plane_indices]
  inds = rand(1 : length(vec(Θ)), 10000000)
  inds_plane = rand(1 : length(vec(Θ_plane)), 1000000)

  Θ_inv_plot = scatter(vec(distance_matrix)[inds], log10.(abs.(vec(Θ_inv[inds]))))
  scatter!(Θ_inv_plot, vec(distance_matrix_plane)[inds_plane], log10.(abs.(vec(Θ_inv_plane[inds_plane]))))
  Θ_plot = scatter(vec(distance_matrix)[inds], log10.(abs.(vec(Θ[inds]))))
  scatter!(Θ_plot, vec(distance_matrix_plane)[inds_plane], log10.(abs.(vec(Θ_plane[inds_plane]))))


  savefig(Θ_inv_plot, "./out/exponential_decay_inv_$(distance)_q_$q.png")
  savefig(Θ_plot, "./out/exponential_decay_$(distance)_q_$q.png")

  max_violation, ind_max_violation = findmax((distance_matrix .> 2.5) .* (abs.(Θ_inv)))

  display(hwv[ind_max_violation.I[1]])
  display(hwv[ind_max_violation.I[2]])
  @show dist(hwv[ind_max_violation.I[1]], hwv[ind_max_violation.I[2]])

end 
