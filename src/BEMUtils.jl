using StaticArrays
# using HCubature
using Cubature
using LinearAlgebra
using Rotations
include("KoLesky.jl")
include("SortSparse.jl")


cubature(f, xmin, xmax) = hcubature(f, xmin, xmax)
# cubature(f, xmin, xmax) = pcubature(f, xmin, xmax)

# Code for creating the matrix element of a simple boundary element method on quadrilateral domains
abstract type AbstractElement{T} end

function Φ(r)
  return 1/ 4 / π / norm(r)
end

# function that gives the integral of the Green's function with respect to the 
# first variable, over a right-angled triangle with sidelenght a and b.
# Here, the singularity is height h above the point where a and c intersect.
function ∫Φ_right_angle(a, b, h)
  r = a/b
  h = abs(h)
  iszero(b) && return b 

  return b * log(sqrt((h^2 + a^2 + b^2) / (h^2 + b^2)) + a / sqrt(h^2 + b^2)) -
  h / 2 * atan(2 * h * sqrt(a^2 + b^2 + h^2), r * (h^2 - b^2) - (h^2 + b^2) / r) +
  h / 2 * atan(2, (r - 1/r))
end

# version for h approximately 0
function ∫Φ_right_angle(a, b)
  r = a/b
  iszero(b) && return b 
  return b * log(1 + r + r^2)
end



# Using the above function for the integral against a general angle triangle
# a⃗, b⃗ displacements to from the projection of the 
# singularity onto to the two other vertices
# h is the distance of the singularity to the plane
function ∫Φ_general_angle(a⃗, b⃗, h)
  if a⃗ ≉ b⃗
    λ = first(b⃗' * b⃗ - a⃗' * b⃗) / first((a⃗ - b⃗)' * (a⃗ - b⃗))
  else  
    λ = eltype(b⃗)(1/2)
  end
  h⃗ = λ * a⃗ + (1 - λ) * b⃗

  return sign(1 - λ) * ∫Φ_right_angle(norm(a⃗ - h⃗), norm(h⃗), h) + 
         sign(λ) * ∫Φ_right_angle(norm(b⃗ - h⃗), norm(h⃗), h)
end

# version for h approximately 0
function ∫Φ_general_angle(a⃗, b⃗)
  if a⃗ ≉ b⃗
    λ = first(b⃗' * b⃗ - a⃗' * b⃗) / first((a⃗ - b⃗)' * (a⃗ - b⃗))
  else  
    λ = eltype(b⃗)((1/2))
  end
  h⃗ = λ * a⃗ + (1 - λ) * b⃗

  return sign(1 - λ) * ∫Φ_right_angle(norm(a⃗ - h⃗), norm(h⃗)) + 
         sign(λ) * ∫Φ_right_angle(norm(b⃗ - h⃗), norm(h⃗))
end


# Function to integrate a Green's function over an axis-aligned square 
# with center $c$ (relative to the projection of the singularity onto the 
# plane described by the square) and diameter $d$. 
function ∫Φ_square(c⃗, d, h)
  return sign(c⃗[1] + d/2) * ∫Φ_general_angle(c⃗ .+ (d/2, d/2),
                                             c⃗ .+ (d/2, -d/2), h) -
         sign(c⃗[1] - d/2) * ∫Φ_general_angle(c⃗ .+ (-d/2, d/2),
                                             c⃗ .+ (-d/2, -d/2), h) +
         sign(c⃗[2] + d/2) * ∫Φ_general_angle(c⃗ .+ (d/2, d/2),
                                             c⃗ .+ (-d/2, d/2), h) -
         sign(c⃗[2] - d/2) * ∫Φ_general_angle(c⃗ .+ (d/2, -d/2),
                                             c⃗ .+ (-d/2, -d/2), h) 
end

# version for h = 0
function ∫Φ_square(c⃗, d)
  return sign(c⃗[1] + d/2) * ∫Φ_general_angle(c⃗ .+ (d/2, d/2),
                                             c⃗ .+ (d/2, -d/2)) -
         sign(c⃗[1] - d/2) * ∫Φ_general_angle(c⃗ .+ (-d/2, d/2),
                                             c⃗ .+ (-d/2, -d/2)) +
         sign(c⃗[2] + d/2) * ∫Φ_general_angle(c⃗ .+ (d/2, d/2),
                                             c⃗ .+ (-d/2, d/2)) -
         sign(c⃗[2] - d/2) * ∫Φ_general_angle(c⃗ .+ (d/2, -d/2),
                                             c⃗ .+ (-d/2, -d/2)) 
end



struct HaarScaling{T}<:AbstractElement{T} 
  v::SVector{3,T} 
  A::SArray{Tuple{3,2},T,2,6}
  ν::SVector{3,T} 
  
  function HaarScaling{T}(v, A) where T
    ν = SVector{3,T}(vec(nullspace(A')))
    ν = ν / norm(ν)
    return new(v, A, ν)
  end
end

function constructMatrixElement_numeric(e1::HaarScaling{RT}, e2::HaarScaling{RT}) where RT
  scalefac::RT = (sqrt(det(e1.A' * e1.A)) * sqrt(det(e2.A' * e2.A)))
  # function to be integrated 
  function tempΦ(arg) 
     return Φ(e1.v - e2.v + 
              e1.A[:,1] * arg[1] + 
              e1.A[:,2] * arg[2] - 
              e2.A[:,1] * arg[3] -
              e2.A[:,2] * arg[4]) * scalefac
  end

  # Compute the cubature
  hc_result = cubature(tempΦ, (-1, -1, -1, -1), (1, 1, 1, 1))
  @show hc_result[2]

  return hc_result[1] / int_scale(e1) / int_scale(e2)
end

function constructMatrixElement(e1::HaarScaling{RT}, e2::HaarScaling{RT}) where RT
  scalefac::RT = sqrt(det(e1.A' * e1.A))
  # function to be integrated 
  function tempΦ(arg) 
     return constructMatrixElement(Point{RT}(e1.v + 
                                             e1.A[:,1] * arg[1] + 
                                             e1.A[:,2] * arg[2]), e2) * scalefac
  end

  # Compute the cubature
  hcout = cubature(tempΦ, (-1, -1), (1, 1))
  return hcout[1] # / int_scale(e1) / int_scale(e2)

end




struct Point{T}<:AbstractElement{T}
  v::SVector{3,T}
end

function Point(v::AbstractVector{RT}) where RT
  return Point{RT}(SVector{3,RT}(v[1], v[2], v[3]))
end

function points(mat::AbstractMatrix{RT}) where RT
  @assert size(mat, 1) == 3
  outArray = Vector{Point{RT}}(undef, size(mat, 2))
  for j = 1 : size(mat, 2)
    outArray[j] = Point{RT}(mat[:,j])
  end
  return outArray
end

function constructMatrixElement(e1::Point{RT}, e2::Point{RT}) where RT
  return RT(Φ(e1.v - e2.v))
end


function constructMatrixElement_numeric(e1::Point{RT}, e2::HaarScaling{RT}) where RT
  @show scalefac = sqrt(det(e2.A' * e2.A))
  function tempΦ(arg) 
     return RT(Φ(e2.v - e1.v + 
                e2.A[:,1] * arg[1] +
                e2.A[:,2] * arg[2]) * scalefac)
  end

  # Compute the cubature
  return cubature(tempΦ, (-1, -1), (1, 1))[1]
end

function constructMatrixElement(e1::Point{RT}, e2::HaarScaling{RT}) where RT
  # Presently only works for square elements
  # Diameter of square
  d = sqrt(sqrt(det(4 * e2.A' * e2.A)))
  c⃗ = e2.v .- e1.v
  # projection onto plane described by element
  Pc⃗ = SVector(dot(e2.A[:,1], c⃗), dot(e2.A[:,2], c⃗)) / (d / 2)
  h = norm(c⃗ - e2.A * Pc⃗ / (d / 2))

  return ∫Φ_square(Pc⃗, d, h) / 4 / π
end


function constructMatrixElement(e2::HaarScaling, e1::Point)
  return constructMatrixElement(e1, e2)
end

struct HaarWavelet{T}<:AbstractElement{T}
  scalingFunctions::SVector{4, HaarScaling{T}}
  weights::SVector{4,T}
  v::SVector{3,T} 
  A::SArray{Tuple{3,2},T,2,6}
  ν::SVector{3,T} 
  # inner constructor
  function HaarWavelet{T}(scalingFunctions, weights) where {T} 
    if length(scalingFunctions) == 1
      v = first(scalingFunctions).v
      A = first(scalingFunctions).A
      ν = first(scalingFunctions).ν
    elseif length(scalingFunctions) == 4
      v = sum(getfield.(scalingFunctions, :v)) / 4
      A = 2 * first(scalingFunctions).A
      ν = first(scalingFunctions).ν
    else 
      throw(DomainError(scalingFunctions, "must be length 1 or 4"))
    end
  return new(scalingFunctions, weights, v, A, ν)
  end
end

# Function to construct a planar Haar scaling function centered in v with lenght scale h
function HaarScaling(h, v)
  RT = eltype(v) 
  return HaarScaling{RT}(v, 
          @SMatrix [ h/2 0   ;
                     0   h/2 ;
                     0   0   ] )
end

# Function to construct a planar Haar wavelet centered in v with length scale h 
# of type i

coefMat = @SMatrix [ 1  1   1   1;
                     1  1   -1  -1;
                     1  -1  -1  1; 
                     1  -1  1   -1 ]

function HaarWavelet(h, v, i)
  RT = eltype(v) 
  return HaarWavelet{RT}(SVector{4,HaarScaling{RT}}(HaarScaling(1/2 * h, v .+ (h/4, h/4, 0)),
                                                    HaarScaling(1/2 * h, v .+ (-h/4, h/4, 0)),
                                                    HaarScaling(1/2 * h, v .+ (h/4, -h/4, 0)),
                                                    HaarScaling(1/2 * h, v .+ (-h/4, -h/4, 0))),
                         SVector{4,RT}((coefMat[1, i], 
                                        coefMat[2, i],
                                        coefMat[3, i],
                                        coefMat[4, i]))) 
end

function constructMatrixElement(e1::Point{RT}, e2::HaarWavelet{RT}) where RT
  out = zero(RT)
  for j = 1 : 4
      out += e2.weights[j] * constructMatrixElement(e1, e2.scalingFunctions[j])
  end
  return out
end

function constructMatrixElement(e2::HaarWavelet, e1::Point)
  return constructMatrixElement(e1, e2)
end


function constructMatrixElement(e1::HaarWavelet{RT}, e2::HaarWavelet{RT}) where RT
  out::RT = zero(RT)
  if scale(e1) > scale(e2)
    tmp = e1
    e1 = e2
    e2 = tmp
  end
  for i = 1 : 4
    for j = 1 : 4
      out += RT(e1.weights[i] * e2.weights[j] * constructMatrixElement(e1.scalingFunctions[i], e2.scalingFunctions[j]))
    end
  end
  return RT(out)
end

# Attempts at combining the evaluations of the haar wavelets, but doesn't seem to help.
# function constructMatrixElement(e1::HaarWavelet{RT}, e2::HaarWavelet{RT}) where RT
#   if scale(e1) > scale(e2)
#     tmp = e1
#     e1 = e2
#     e2 = tmp
#   end
#   scalefac::RT = sqrt(det(e1.scalingFunctions[1].A' * e1.scalingFunctions[1].A))
# 
#   function tempΦ(arg) 
#     out = zero(RT)
#     for i = 1 : 4
#       for j = 1 : 4
#         out += e1.weights[i] * e2.weights[j] * scalefac * 
#                constructMatrixElement(Point{RT}(e1.scalingFunctions[i].v + 
#                                                 e1.scalingFunctions[i].A[:,1] * arg[1] + 
#                                                 e1.scalingFunctions[i].A[:,2] * arg[2]), e2.scalingFunctions[j]) 
#       end
#     end
#     return out
#   end
# 
#   # Compute the cubature
#   hcout = cubature(tempΦ, (-1, -1), (1, 1))
#   return hcout[1]
# end


function scale(e::HaarScaling{RT}) where RT
  return RT(sqrt(sqrt(det(e.A' * e.A))) * 2)
end

function scale(e::HaarWavelet{RT}) where RT
  return RT(sqrt(sqrt(det(e.A' * e.A))) * 2)
end

function int_scale(e::Point{RT}) where RT
  return one(RT)
end

function int_scale(e::HaarScaling)
  return scale(e)
end

function int_scale(e::HaarWavelet)
  return scale(first(e.scalingFunctions))
end

function center(e::HaarWavelet)
  return sum(getfield.(e.scalingFunctions, :v))/4
end

# Implement translation of basis functions 
function translate(e::HaarScaling{RT}, v) where RT
  return HaarScaling{RT}(e.v .+ v, e.A)
end

function translate(e::HaarWavelet{RT}, v) where RT
  return HaarWavelet{RT}(broadcast(es -> translate(es, v), e.scalingFunctions), e.weights)
end

function translate(eVector::Vector{HaarWavelet{RT}}, v) where {RT}
  return broadcast(es -> translate(es, v), eVector)
end

# Implement rotation (or more general linear transform) of basis functions
function rotate(e::HaarScaling{RT}, O) where RT
  return HaarScaling{RT}(O * e.v, O * e.A)
end

function rotate(e::HaarWavelet{RT}, O) where RT
  return HaarWavelet{RT}(broadcast(es -> rotate(es, O), e.scalingFunctions), e.weights)
end

function rotate(eVector::Vector{HaarWavelet{RT}}, O) where {RT}
  return broadcast(es -> rotate(es, O), eVector)
end



# Function to construct Haar wavelets of level q on the surface of the cube 
function haarCube(q) 
  RT = Float64
  haarArray = Vector{HaarWavelet{RT}}(undef, 4^q)
  ofs = 0
  for k = 1 : q
    if k == 1 
      for type = 1 : 4
        ofs += 1
        haarArray[ofs] = HaarWavelet(1.0, (0.5,0.5,0.0), type)
      end
    else
      for (i,j) in Iterators.product(1 : 2^(k-1), 1:2^(k-1))  
        coordOffset = -1/2^(k)
        coordScale = 1/2^(k-1)
        for type = 2 : 4
          ofs += 1
          haarArray[ofs] = HaarWavelet(coordScale, coordOffset .+ (i * coordScale, j * coordScale, -coordOffset), type)
        end
      end
    end
  end

  #Add translated and rotated versions to construct the full cube
  haarArray = vcat(haarArray, translate(haarArray, (0,0,1)),
                   rotate(haarArray, RotX(π/2)), translate(rotate(haarArray, RotX(π/2)), (0,1,0)),
                   rotate(haarArray, RotY(-π/2)), translate(rotate(haarArray, RotY(-π/2)), (1,0,0)))
  return haarArray
end

import Base.show
function show(io::IO, hs::HaarScaling) 
  println("v  = $(hs.v)")
  println("b1 = $(hs.A[:,1])")
  println("b2 = $(hs.A[:,2])")
  println("ν  = $(hs.ν)")
end

function show(io::IO, hw::HaarWavelet) 
  println("v  = $(hw.v)")
  println("b1 = $(hw.A[:,1])")
  println("b2 = $(hw.A[:,2])")
  println("ν  = $(hw.ν)")

  println("Weights = $(hw.weights)")
  println("Scaling functions =")
  # for i = 1 : 4
  #   display(hw.scalingFunctions[i])
  # end
end

function assembleMatrix(eList1::AbstractVector{ET1}, eList2::AbstractVector{ET2}) where {ET1,ET2 <:AbstractElement{RT}} where RT
  outMatrix = Matrix{RT}(undef, length(eList1), length(eList2))  
  @time begin
    Threads.@threads for i = 1 : length(eList1)
      # @show i
      for j = 1 : length(eList2)
        # @show j
        outMatrix[i,j] = constructMatrixElement(eList1[i], eList2[j])
      end
    end
  end
  return outMatrix
end

function assembleMatrix(eList::AbstractVector{ET}) where {ET <:AbstractElement{RT}} where RT
  outMatrix = Matrix{RT}(undef, length(eList), length(eList))  
  @time begin
    Threads.@threads for i = 1 : length(eList)
      for j = 1 : i
        outMatrix[i,j] = constructMatrixElement(eList[i], eList[j])
        outMatrix[j,i] = outMatrix[i,j]
      end
    end
  end
  return outMatrix
end


function evaluateSolution(x::AbstractVector{RT}, elements::AbstractVector{ET}, weights::AbstractVector{RT}) where ET<:AbstractElement{RT} where RT
  out = zero(RT)
  @assert length(elements) == length(weights)
  for (i, e) in enumerate(elements)
    out += constructMatrixElement(point(x), e) * weights[i]
  end
  return out
end

function dist_sphere(e1::HaarWavelet, e2::HaarWavelet)
  sc1 = scale(e1) 
  sc2 = scale(e2)
  return max(0.0, norm(e1.v - e2.v) - sc1/sqrt(2) - sc2/sqrt(2))
end

function dist(e1::HaarWavelet, e2::HaarWavelet)
  # make sure first element is always the smaller one
  sc1 = scale(e1)
  sc2 = scale(e2)
  if sc1 > sc2
    tmpsc = sc1
    tmp = e1
    e1 = e2
    sc1 = sc2
    e2 = tmp
    sc2 = tmpsc
  end


  # If it is possible that one of the wavelets is contained in the other
  if sc1 < sc2 && 
     (e1.ν ≈ e2.ν || e1.ν ≈ - e2.ν) && 
     isapprox(dot(e1.ν, e1.v - e2.v) / norm(e1.v - e2.v), zero(eltype(e1.v)), atol = 1e-6) && 
     maximum(abs.(e2.A' * (e1.v - e2.v))) ≤ (sc2/2)^2
    
    return zero(eltype(e1.v))
  end

  out = typemax(eltype(e1.v))
  for v1 = @SVector [SVector(1,1), SVector(-1,1), SVector(1,-1), SVector(-1,-1)]
    for v2 = @SVector [SVector(1,1), SVector(-1,1), SVector(1,-1), SVector(-1,-1)]
      out = min(out, norm(e1.v + e1.A * v1 - 
                          (e2.v + e2.A * v2)))
    end
  end
  return out
end

function sortSparseRev(hwl::AbstractVector{HaarWavelet{RT}}, ρ::RT) where RT
  # ordering the array, in case it hasn;t been ordered yet
  N = length(hwl)
  P = sortperm(scale.(hwl))
  hwl = hwl[P]
  revP = similar(P)
  revP[P] = 1:N
  distances = scale.(hwl)

  n_entries = N
  counter = 1
  I = Vector{Int}(undef, n_entries)
  J = Vector{Int}(undef, n_entries)

  for i = 1 : N
    for j = 1 : i
      if dist_sphere(hwl[i], hwl[j]) ≤ ρ * min(distances[i], distances[j])
        I[counter] = i
        J[counter] = j
        
        if counter == n_entries
          n_entries += N
          resize!(I, n_entries)
          resize!(J, n_entries)
        end
        counter += 1
      end
    end
  end
  resize!(I, counter - 1)
  resize!(J, counter - 1)

  L = sparse(I, J, ones(length(I)))
  return L.colptr, L.rowval, P, revP, distances
end

# creates the Kij! function when given train and test x containing of 
# abstract elements 
function create_Kij!_closure(
          skeletons::Vector{Skeleton{Ti}}, 
          xTrain::AbstractVector{<:AbstractElement}, 
          xTest::AbstractVector{<:AbstractElement},
          σ=zeros(length(xTrain))) where Ti<:Integer

  NTrain = length(xTrain)
  NTest = length(xTest)
  N = NTrain + NTest

  # first taking care of the interactions between boundary elements
  n_entries = N
  counter = 1
  I = Vector{Ti}(undef, n_entries)
  J = Vector{Ti}(undef, n_entries)

  for skel in skeletons
    for j in skel.parents
      for i in skel.children
        I[counter] = i
        J[counter] = j

        if counter == n_entries
          n_entries += N
          resize!(I, n_entries)
          resize!(J, n_entries)
        end
        counter += 1
      end
    end
  end

  resize!(I, counter - 1)
  resize!(J, counter - 1)
  L = tril(sparse(I, J, zeros(length(I))))
  I, J, S = findnz(L)

  @show 2 * length(S) / N / N 

  # Computing the nonzero entries corresponding to 3 
  # interactions among boundary panels
  @time Threads.@threads for k = 1 : length(I)
    S[k] = constructMatrixElement(xTrain[I[k]], xTrain[J[k]])
    if I[k] == J[k]
      S[k] += σ[I[k]]
    end
  end

  # computing extending the index sets to account for the 
  # interaction between boundary points and test points 
  nnz_boundary = length(I)
  resize!(I, nnz_boundary + length(xTest) * length(xTrain) + length(xTest))
  resize!(J, nnz_boundary + length(xTest) * length(xTrain) + length(xTest))
  resize!(S, nnz_boundary + length(xTest) * length(xTrain) + length(xTest))

  counter = nnz_boundary + 1
  for i = 1 : length(xTest)
    for j = 1 : length(xTrain)
      I[counter] = length(xTrain) + i
      J[counter] = j
      counter += 1
    end
  end

  # Computing interactions between boundary panels and test 
  # points.
  Threads.@threads for k = 1 : (length(xTest) * length(xTrain))
    S[nnz_boundary + k] = constructMatrixElement(xTest[I[nnz_boundary + k] - length(xTrain)], xTrain[J[nnz_boundary + k]])
  end

  I[(end - length(xTest) + 1) : end] .= (length(xTrain) + 1 ) : (length(xTrain) + length(xTest))
  J[(end - length(xTest) + 1) : end] .= (length(xTrain) + 1 ) : (length(xTrain) + length(xTest))
  S[(end - length(xTest) + 1) : end] .= 1e30

  Θ::Hermitian{Float64,SparseMatrixCSC{Float64,Ti}} = Hermitian(sparse(I, J, S), :L)

  # A function that fills a matrix with the entries of the covariance matrix corresponding to a given set of indices
  function Kij!(Out::AbstractArray, 
                I::AbstractVector,
                J::AbstractVector)
    # vec(Out) .= vec(Θ[I, J])
    for (ind_i, i) in enumerate(I)
      for (ind_j, j) in enumerate(J)
        Out[ind_i, ind_j] = Θ[i, j]
      end
    end
  end
  return Kij!
end

# creates the Kij! function when given train and test x containing 
# abstract elements 
function create_Kij!_closure(
          skeletons::Vector{Skeleton{Ti}}, 
          xTrain::AbstractVector{<:AbstractElement}, 
          xTest::AbstractVector{<:AbstractElement},
          Θ_dense::AbstractMatrix,
          σ=zeros(length(xTrain))) where Ti<:Integer

  NTrain = length(xTrain)
  NTest = length(xTest)
  N = NTrain + NTest

  # first taking care of the interactions between boundary elements
  n_entries = N
  counter = 1
  I = Vector{Ti}(undef, n_entries)
  J = Vector{Ti}(undef, n_entries)

  for skel in skeletons
    for j in skel.parents
      for i in skel.children
        I[counter] = i
        J[counter] = j

        if counter == n_entries
          n_entries += N
          resize!(I, n_entries)
          resize!(J, n_entries)
        end
        counter += 1
      end
    end
  end

  resize!(I, counter - 1)
  resize!(J, counter - 1)
  L = tril(sparse(I, J, zeros(length(I))))
  I, J, S = findnz(L)

  @show 2 * length(S) / N / N 

  # Computing the nonzero entries corresponding to 3 
  # interactions among boundary panels
  @time Threads.@threads for k = 1 : length(I)
    S[k] = Θ_dense[I[k], J[k]]
    if I[k] == J[k]
      S[k] += σ[I[k]]
    end
  end

  # computing extending the index sets to account for the 
  # interaction between boundary points and test points 
  nnz_boundary = length(I)
  resize!(I, nnz_boundary + length(xTest) * length(xTrain) + length(xTest))
  resize!(J, nnz_boundary + length(xTest) * length(xTrain) + length(xTest))
  resize!(S, nnz_boundary + length(xTest) * length(xTrain) + length(xTest))

  counter = nnz_boundary + 1
  for i = 1 : length(xTest)
    for j = 1 : length(xTrain)
      I[counter] = length(xTrain) + i
      J[counter] = j
      counter += 1
    end
  end

  # Computing interactions between boundary panels and test 
  # points.
  Threads.@threads for k = 1 : (length(xTest) * length(xTrain))
    S[nnz_boundary + k] = constructMatrixElement(xTest[I[nnz_boundary + k] - length(xTrain)], xTrain[J[nnz_boundary + k]])
  end

  I[(end - length(xTest) + 1) : end] .= (length(xTrain) + 1 ) : (length(xTrain) + length(xTest))
  J[(end - length(xTest) + 1) : end] .= (length(xTrain) + 1 ) : (length(xTrain) + length(xTest))
  S[(end - length(xTest) + 1) : end] .= 1e30

  Θ::Hermitian{Float64,SparseMatrixCSC{Float64,Ti}} = Hermitian(sparse(I, J, S), :L)

  # A function that fills a matrix with the entries of the covariance matrix corresponding to a given set of indices
  function Kij!(Out::AbstractArray, 
                I::AbstractVector,
                J::AbstractVector)
    # vec(Out) .= vec(Θ[I, J])
    for (ind_i, i) in enumerate(I)
      for (ind_j, j) in enumerate(J)
        Out[ind_i, ind_j] = Θ[i, j]
      end
    end
  end
  return Kij!
end

# creates the Kij! function when given train and test x containing 
# abstract elements from a precomputed matrix, without including the 
# interactions between boundary and interior
function create_Kij!_closure(
          skeletons::Vector{Skeleton{Ti}}, 
          xTrain::AbstractVector{<:AbstractElement}, 
          Θ_dense::AbstractMatrix,
          σ=zeros(length(xTrain))) where Ti<:Integer

  NTrain = length(xTrain)
  N = NTrain 

  # first taking care of the interactions between boundary elements
  n_entries = N
  counter = 1
  I = Vector{Ti}(undef, n_entries)
  J = Vector{Ti}(undef, n_entries)

  for skel in skeletons
    for j in skel.parents
      for i in skel.children
        I[counter] = i
        J[counter] = j

        if counter == n_entries
          n_entries += N
          resize!(I, n_entries)
          resize!(J, n_entries)
        end
        counter += 1
      end
    end
  end

  resize!(I, counter - 1)
  resize!(J, counter - 1)
  L = tril(sparse(I, J, zeros(length(I))))
  I, J, S = findnz(L)

  @show 2 * length(S) / N / N 

  # Computing the nonzero entries corresponding to 3 
  # interactions among boundary panels
  @time Threads.@threads for k = 1 : length(I)
    S[k] = Θ_dense[I[k], J[k]]
    if I[k] == J[k]
      S[k] += σ[I[k]]
    end
  end

  Θ::Hermitian{Float64,SparseMatrixCSC{Float64,Ti}} = Hermitian(sparse(I, J, S), :L)

  # A function that fills a matrix with the entries of the covariance matrix corresponding to a given set of indices
  function Kij!(Out::AbstractArray, 
                I::AbstractVector,
                J::AbstractVector)
    # vec(Out) .= vec(Θ[I, J])
    for (ind_i, i) in enumerate(I)
      for (ind_j, j) in enumerate(J)
        Out[ind_i, ind_j] = Θ[i, j]
      end
    end
  end
  return Kij!
end


# Version of the predict function that takes in haar wavelets
# It precomputes all matrix entries needed, in order to minimize the
# amount of numerical integration.
function predict(skeletons::Vector{Skeleton{Ti}}, 
                 yTrain, 
                 xTrain::AbstractVector{<:AbstractElement}, 
                 xTest::AbstractVector{<:AbstractElement},
                 σ=zeros(length(xTrain))) where {Ti}
    NTrain = length(xTrain)
    NTest = length(xTest)
    N = NTrain + NTest

    Kij! = create_Kij!_closure(skeletons, xTrain, xTest, σ) 
    ITest = UnitRange{Ti}(NTrain + 1, N)
    return μ, σ_posterior = predict(skeletons, ITest, yTrain, Kij!)
end

# Version of the predict function takes in a matrix that contains all the matrix elements
function predict(skeletons::Vector{Skeleton{Ti}}, 
                 yTrain, 
                 xTrain::AbstractVector{<:AbstractElement}, 
                 xTest::AbstractVector{<:AbstractElement},
                 Θ_dense::AbstractMatrix,
                 σ=zeros(length(xTrain))) where {Ti}
    NTrain = length(xTrain)
    NTest = length(xTest)
    N = NTrain + NTest

    Kij! = create_Kij!_closure(skeletons, xTrain, xTest, Θ_dense, σ)
    ITest = UnitRange{Ti}(NTrain + 1, N)
    return μ, σ_posterior = predict(skeletons, ITest, yTrain, Kij!)
end

function assembleL(skeletons::Vector{Skeleton{Ti}}, x::AbstractVector{<:AbstractElement}, Θ_dense::AbstractMatrix, σ=zeros(length(x))) where {Ti}
    # A function that fills a matrix with the entries of the covariance matrix corresponding to a given set of indices
    Kij! = create_Kij!_closure(skeletons, x, Θ_dense, σ)

    # Computing the required size of the buffers 
    maxChildren = maximum(length.( getfield.(skeletons, :children)))
    maxParents = maximum(length.( getfield.(skeletons, :parents)))

    # Preallocating output (slightly excessive ram usage since only approximately lower triangular)
    maxEntries = sum([length(s.children) * length(s.parents) for s in skeletons])
    I = Vector{Ti}(undef, maxEntries)
    J = Vector{Ti}(undef, maxEntries)
    S = Vector{Float64}(undef, maxEntries)
    offset = 0

    
    #Preallocating buffers 
    LBuffers = Vector{Vector{Float64}}(undef, Threads.nthreads())
    BBuffers = Vector{Vector{Float64}}(undef, Threads.nthreads())
    for k = 1 : Threads.nthreads()
      LBuffers[k] = Vector{Float64}(undef, maxChildren^2)
      BBuffers[k] = Vector{Float64}(undef, maxChildren * maxParents)
    end

    # creates 
    offsets = vcat(0, cumsum([length(s.children) * length(s.parents) for s in skeletons])[1 : (end - 1)])

    LinearAlgebra.BLAS.set_num_threads(1)

    Threads.@threads for k = 1 : length(skeletons)
        skel = skeletons[k]
        nChildren = length(skel.children)
        nParents = length(skel.parents)
        # Setting up the outputs:
        L = unsafe_wrap(Array{Float64,2}, pointer(LBuffers[Threads.threadid()]), (nChildren, nChildren))
        B = unsafe_wrap(Array{Float64,2}, pointer(BBuffers[Threads.threadid()]), (nChildren, nParents))

            
        # Do the prediction using the io variables defined above
        assembleLSkeleton!(L, B, skel, Kij!)
        for (l, tup) in enumerate(Iterators.product(skel.children, skel.parents))
            I[offsets[k] + l] = tup[1]
            J[offsets[k] + l] = tup[2]
            S[offsets[k] + l] = B[l] 
        end
    end

    # Selecting the lower triangular part
    inds = findall(I .>= J)
    I, J, S = I[inds], J[inds], S[inds]

    # Construct sparse matrix and return
    return sparse(I,J,S)
end
