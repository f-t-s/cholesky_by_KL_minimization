include("MutHeap.jl")
using LinearAlgebra
using SparseArrays
using Statistics
#############################################################################
#Introducing the "daycare", which keeps track of the descendants of every node
#The struct is essentially a buffered lower triangular CSC sparse matrix,
#together with an of the degrees of freedom
#############################################################################
struct Member{Tval, Tid}
  val::Tval
  id::Tid
end

import Base.isless
function isless(a::Member{Tval, Tid}, b::Member{Tval, Tid}) where{Tval, Tid}
  return isless((a.val, a.id), (b.val, b.id))
end



mutable struct ChildList{Tval, Tid}
  NParents::Tid
  NChildren::Tid
  NBuffer::Tid

  #This array gives for contains the ordering. The i-th parent in the 
  #daycare has id P[i]
  P::Vector{Tid}
  #This array contains as the i-th element the number that the ith parent 
  #has with respect to the multiresolution ordering.
  revP::Vector{Tid}
  
  #The array that contains the first "child" for every parent
  colptr::Vector{Tid}

  #The array that contains the global id-s of the children 
  rowval::Vector{Member{Tval,Tid}}
end


#Function that begins a new parent aka column in daycare
function newParent(dc::ChildList, IdParent)
  dc.NParents += 1 
  dc.P[ dc.NParents ] = IdParent
  dc.colptr[ dc.NParents ] = dc.NChildren + 1
  dc.colptr[ dc.NParents + 1 ] = dc.NChildren + 1
  dc.revP[ IdParent ] = dc.NParents
end

function newChild(dc::ChildList, newChildId) 
  # If the buffer is flowing over, increase it
  if dc.NChildren >= dc.NBuffer
    if dc.NChildren <= 1e6
      dc.NBuffer = 2 * dc.NBuffer 
    else 
      dc.NBuffer = dc.NBuffer + 1e6
    end
    resize!( dc.rowval, dc.NBuffer )
  end
  dc.NChildren += 1
  dc.colptr[dc.NParents + 1] += 1
  dc.rowval[dc.NChildren] = newChildId 
end

function newChildren( dc::ChildList, newChildren)
  # If the buffer is flowing over, increase it
  while dc.NChildren + size(newChildren,1) >= dc.NBuffer - 1
    if dc.NChildren <= 1e6
      dc.NBuffer = 2 * dc.NBuffer 
    else 
      dc.NBuffer = dc.NBuffer + 1e6
    end
    resize!( dc.rowval, dc.NBuffer )
  end

  dc.NChildren += size(newChildren,1)
  dc.colptr[dc.NParents + 1] += size(newChildren,1)
  dc.rowval[dc.NChildren - size(newChildren,1) + 1 : dc.NChildren] .= newChildren
end

function _determineChildren!(h::MutHeap,
                             dc::ChildList,
                             parents::Vector,
                             pivot::Node,
                             buffer,
                             rho,
                             dist2Func)
#Function to determine the children of a node in the ordering and sparsity
#pattern.
#TODO Update description
#Inputs:
# h:
#   Heap to keep track of the elements and their distance to the points added
#   already.
# dc: 
#   The "daycare" keeping track of the nodes and their children.
# parents:
#   Array that in its i-th position contains the a node with the id of the 
#   preferred parent of the i-th node and the distance of the i-th node to 
#   its preferred parent.
# Id: 
#   The Id of the point, the children of which are being determined.
# rho:
#   The oversampling parameter determining the size of the sparsity pattern.
# dist:
#   The function dist(i,j) gives the distance between the points with ids 
#   i and j.

  #adding the new parent
  distToParent = parents[pivot.id].val
  lengthscale = pivot.val
  iterBuffer = 0

  for index = dc.colptr[dc.revP[parents[pivot.id].id]] : (dc.colptr[
                    dc.revP[parents[pivot.id].id] + 1] - 1)
    #The candidate point for the pivots children
    candidate = dc.rowval[index]
    #Distance of the candidate to the pivot:
    dist2 = dist2Func(candidate.id, pivot.id)
    #Check whether the candidate is close enough to be added as a child
    if iszero(dc.revP[candidate.id]) && (dist2 <= (lengthscale * rho)^2)
      dist = sqrt(dist2)
      #Step 1: We add a new child to the pivot:
      #Increase the count of added children by one
      iterBuffer += 1
      #Add node representing the new child to buffer
      buffer[iterBuffer] = Member(dist, candidate.id)
      #Step 2: We possibly update the parent update the distance of the point 
      newDist = update!(h, candidate.id, dist)
      #Step 3: If possible and useful, update the preferred parent:
      if (dist + rho * newDist <= rho * lengthscale) && 
         (dist < parents[candidate.id].val)  
         parents[candidate.id] = Member(dist,pivot.id)
      end
    #If the candidate is far enough away from the pivots parent, such that it can
    #not possibly be within reach, break:
    elseif candidate.val > distToParent + lengthscale * rho
      break
    end
  end
  viewBuffer = view(buffer, 1 : iterBuffer)
  sort!(viewBuffer, alg=QuickSort)
  newParent(dc, pivot.id)
  newChildren(dc, viewBuffer)
end

#Function that constructs the ordering and sparsity pattern from the
#a function evaluating the squared distance
function sortSparse(N::Ti, 
                    rho::Tv,
                    dist2Func,
                    initInd = one(Ti)) where {Ti<:Integer, Tv<:Real}
  #Constructing the heap and initialising all variables have maximal distance
  h = MutHeap{Tv,Ti,Ti}(Vector{Node{Tv,Ti,Ti}}(undef, N), Ti(1) : N)
  for i = Ti(1) : N
    h.nodes[i] = Node(typemax(Tv), i, zero(Ti))
  end
  #Constructing the Daycare The permutation matrices are initialized to the 
  #maximal element to force errors in case an index is being used before it
  #has been initialized.
  dc = ChildList{Tv,Ti}(zero(Ti), 
        zero(Ti),
        N,
        zeros(Ti,N),
        zeros(Ti,N),
        zeros(Ti, N + one(Ti)),
        Vector{Member{Tv,Ti}}(undef, N))

  #Initializing the Buffer used by _determineChildren!
  nodeBuffer = Vector{Member{Tv,Ti}}(undef, N)

  #Initializing the array that will hold the distances measured in the end
  distances = -ones(Tv, N)

  # Performing the first step of the algorithm:
  # Adding the note initInd as the first parent and making all other nodes its
  # children, as well as updating their distance:
  newParent(dc, initInd)
  # adapting the rank of the node to never be picked again
  h.nodes[initInd] = eltype(h.nodes)(h.nodes[initInd].val,
                                     h.nodes[initInd].id,
                                     typemax(Ti))
  distances[1] = typemax(Tv)
  for i = Ti(1) : N
    #adds a new Child and updates the corresponding distance of the child
    nodeBuffer[i] = 
      Member{Tv,Ti}(update!(h,i,sqrt(dist2Func(i,initInd))),i)
  end
  viewBuffer = view(nodeBuffer, 1 : N)
  sort!(viewBuffer, alg=QuickSort)
  newChildren(dc, viewBuffer)

  # Initialize the buffer keeping track of all the parents
  parents = Vector{Member{Tv,Ti}}(undef, N)
  for i = Ti(1) : N
    parents[i] = Member{Tv,Ti}(sqrt(dist2Func(initInd,i)),initInd)
  end

  for i = Ti(2) : N 
    distances[i] = topNode!(h).val
    _determineChildren!(h,dc,parents,topNode(h),nodeBuffer,rho,dist2Func)
  end

  dc.rowval = dc.rowval[1 : (dc.colptr[end] - 1)]
  
  for k = 1 : size(dc.rowval,1) 
    dc.rowval[k] = Member{Tv,Ti}(dc.rowval[k].val, dc.revP[dc.rowval[k].id])
  end

  return dc.colptr, dc.rowval, dc.P, dc.revP, distances
end

# Function that constructs the ordering and sparsity pattern from the
# a function evaluating the squared distance.
# This function orders the training points before the test points.
function sortSparse(NTrain::Ti, 
                    NTest::Ti, 
                    rho::Tv,
                    dist2Func,
                    initInd = one(Ti)) where {Ti<:Integer, Tv<:Real}
  N = NTrain + NTest
  # Constructing the heap and initialising all variables have maximal distance
  h = MutHeap{Tv,Ti,Ti}(Vector{Node{Tv,Ti,Ti}}(undef, N), Ti(1) : N)
  # Setting the training points with lower rank (appearing first in the ordering)
  for i = Ti(1) : NTrain
    h.nodes[i] = Node(typemax(Tv), i, zero(Ti))
  end
  # Setting the remaining (test) points with higher rank, forcing their appearence later in the ordering.
  for i = (NTrain + Ti(1)) : N
    h.nodes[i] = Node(typemax(Tv), i, one(Ti))
  end

  #Constructing the Daycare The permutation matrices are initialized to the 
  #maximal element to force errors in case an index is being used before it
  #has been initialized.
  dc = ChildList{Tv,Ti}(zero(Ti), 
        zero(Ti),
        N,
        zeros(Ti,N),
        zeros(Ti,N),
        zeros(Ti, N + one(Ti)),
        Vector{Member{Tv,Ti}}(undef, N))

  #Initializing the Buffer used by _determineChildren!
  nodeBuffer = Vector{Member{Tv,Ti}}(undef, N)

  #Initializing the array that will hold the distances measured in the end
  distances = -ones(Tv, N)

  # Performing the first step of the algorithm:
  # Adding the note initInd as the first parent and making all other nodes its
  # children, as well as updating their distance:
  newParent(dc, initInd)
  # adapting the rank of the node to never be picked again
  h.nodes[initInd] = eltype(h.nodes)(h.nodes[initInd].val,
                                     h.nodes[initInd].id,
                                     typemax(Ti))
  distances[1] = typemax(Tv)
  for i = Ti(1) : N
    #adds a new Child and updates the corresponding distance of the child
    nodeBuffer[i] = 
      Member{Tv,Ti}(update!(h,i,sqrt(dist2Func(i,initInd))),i)
  end
  viewBuffer = view(nodeBuffer, 1 : N)
  sort!(viewBuffer, alg=QuickSort)
  newChildren(dc, viewBuffer)

  # Initialize the buffer keeping track of all the parents
  parents = Vector{Member{Tv,Ti}}(undef, N)
  for i = Ti(1) : N
    parents[i] = Member{Tv,Ti}(sqrt(dist2Func(initInd,i)),initInd)
  end

  for i = Ti(2) : N 
    distances[i] = topNode!(h).val
    _determineChildren!(h,dc,parents,topNode(h),nodeBuffer,rho,dist2Func)
  end

  dc.rowval = dc.rowval[1 : (dc.colptr[end] - 1)]
  
  for k = 1 : size(dc.rowval,1) 
    dc.rowval[k] = Member{Tv,Ti}(dc.rowval[k].val, dc.revP[dc.rowval[k].id])
  end

  return dc.colptr, dc.rowval, dc.P, dc.revP, distances
end


#Function constructing the reverse ordering and the minimal distance
#sparsity pattern for the same input data as SortSparse
function sortSparseRev(x::Matrix{Tv},
                       rho::Tv,
                       initInd::Ti) where {Tv<:Real,Ti<:Integer}
  N = size(x,2)
  d = size(x,1)
  #Recast as static arrays to use fast methods provided by StaticArrays.jl
  #Possibly remove, doesn't seem to yield a lot.

  function dist2Func( i::Ti, j::Ti )
    out = zero(Tv)
    @fastmath @inbounds @simd for k = 1 : d
      out += ( x[k,i] - x[k,j] )^2
    end
    return out
  end


  colptr, rowval, P, revP, distances = 
    sortSparse(N, rho, dist2Func, initInd )

  rowvalOut::Vector{Int} = getfield.(rowval, :id)

  I, J, V  = findnz( SparseMatrixCSC{Tv,Ti}(N,N,
                                            colptr, 
                                            rowvalOut, 
                                            ones(Tv, size(rowvalOut,1))) )

  for k = 1 : size( I, 1 ) 
    if sqrt(dist2Func(P[I[k]],P[J[k]])) > rho * min( distances[I[k]], distances[J[k]])
      V[k] = 0.0
    end
  end
  I = (N:-1:1)[I]
  J = (N:-1:1)[J]

  L = sparse( J, I, V )
  dropzeros!(L)

  P = P[N:-1:1]
  revP[P] = 1:N 

  distances = distances[N:-1:1]

  return L.colptr, L.rowval, P, revP, distances
end

#Function constructing the reverse ordering and the minimal distance
#sparsity pattern for the same input data as SortSparse
function sortSparseRev(xTrain::Matrix{Tv},
                       xTest::Matrix{Tv},
                       rho::Tv,
                       initInd::Ti) where {Tv<:Real,Ti<:Integer}
  NTrain = size(xTrain, 2)
  NTest = size(xTest, 2)
  x = hcat(xTrain, xTest)
  N = size(x,2)
  d = size(x,1)

  function dist2Func( i::Ti, j::Ti )
    out = zero(Tv)
    @fastmath @inbounds @simd for k = 1 : d
      out += ( x[k,i] - x[k,j] )^2
    end
    return out
  end


  colptr, rowval, P, revP, distances = 
    sortSparse(NTrain, NTest, rho, dist2Func, initInd)


  rowvalOut::Vector{Ti}  = getfield.(rowval, :id)

  LTemp = SparseMatrixCSC{Tv,Ti}(N,N,
                                 colptr, 
                                 rowvalOut, 
                                 ones(Tv, size(rowvalOut,1)))


  I, J, V  = findnz( SparseMatrixCSC{Tv,Ti}(N,N,
                                            colptr, 
                                            rowvalOut, 
                                            ones(Tv, size(rowvalOut,1))) )


  for k = 1 : size( I, 1 ) 
    if sqrt(dist2Func(P[I[k]],P[J[k]])) > rho * min( distances[I[k]], distances[J[k]])
      V[k] = 0.0
    end
  end
  I = (N:-1:1)[I]
  J = (N:-1:1)[J]

  L = sparse( J, I, V )
  dropzeros!(L)

  P = P[N:-1:1]
  revP[P] = 1:N 

  distances = distances[N:-1:1]

  return L.colptr, L.rowval, P, revP, distances
end

struct Skeleton{Ti}
  parents::Vector{Ti}
  children::Vector{Ti}
end

function cost_storage( s::Skeleton )
  return length(s.parents) * length(s.children)
end

#groupedunique as in base, but without the final resize.
#returns the number of unique elements
function groupedunique!(A::AbstractVector)
    isempty(A) && return A
    idxs = eachindex(A)
    y = first(A)
    # We always keep the first element
    it = iterate(idxs, iterate(idxs)[2])
    count = 1
    for x in Iterators.drop(A, 1)
        if !isequal(x, y)
            y = A[it[1]] = x
            count += 1
            it = iterate(idxs, it[2])
        end
    end
    return count
end


function construct_skeletons( colptr::AbstractVector{Ti},
                              rowval::AbstractVector{Ti},
                              distances::AbstractVector,
                              λ = 3/2 ) where {Ti<:Integer}
  N = length(distances)
  skeletons = Vector{Skeleton{Ti}}(undef, zero(Ti))
  cscratch = Vector{Ti}(undef, 2 * N)
  pscratch = Vector{Ti}(undef, N)
  added = Vector{Bool}(undef, N)
  added .= false

  for k = one(Ti) : N
    #make sure the node had not been added to supernode before
    if added[k] == false 
      pcount = zero(Ti)
      ccount = zero(Ti)
      for l = colptr[k] : (colptr[k+1] - 1) 
        #If the distance criterion is satisfied:
        if (distances[rowval[l]] <= λ * distances[k]) && added[rowval[l]] == false
          #adding the new parent:
          pcount += one(Ti)
          pscratch[pcount] = rowval[l]
          #don't add the same node to another supernode:
          added[rowval[l]] = true  
          #adding the new children:
          increment = colptr[rowval[l] + 1] - colptr[rowval[l]] 
          #If the scratch space is depleted, distill the unique elements
          if ccount + increment > length(cscratch)
            sort!(view(cscratch, 1 : ccount))
            ccount = groupedunique!(view(cscratch, 1 : ccount))
          end
          #Add the new children to the scratch spcae
          cscratch[(ccount + 1) : (ccount + increment)] .= 
            rowval[colptr[rowval[l]] : (colptr[rowval[l] + 1] - 1)]
          #update the ccount:
          ccount += increment
        end
      end
    #globally sort all the children added to the supernode (the parents are 
    #automatically sorted)
    sort!(view(cscratch, 1 : ccount))
    ccount = groupedunique!(view(cscratch, 1 : ccount))
    #append the new supernode
    parents = pscratch[1:pcount]
    children = cscratch[1:ccount]
    push!(skeletons, Skeleton(parents,children))
    end
  end
  return skeletons
end

# Debugging only:
# Function to take a given set of skeletons that 
# returns a new set of skeletons that implement the 
# same sparsity pattern, but with each supernode
# containing only one parent
function singleParents( skeletons::AbstractVector{SkType}) where SkType <: Skeleton 
  n_skeletons = 
    sum([length(s.parents) for s in skeletons])
  out = Vector{SkType}(undef, n_skeletons)
  count = 1
  for s in  skeletons
    for par in s.parents
      k = findfirst(x -> x == par, s.children)
      out[count] = SkType([par], s.children[k:end])
      count += 1
    end
  end
  return out
end

function quadcube(s)
  a = 10.0
  b = 1.0
  c = 1.0
  return (a + b * length(s.parents) + c * length(s.children))length(s.children)^2 
end

# A function to return a partition of the list of skeletons that will have  
# roughly similar work
function partitionSkeletons(work, skeletons, chunksize)
  # model(x, p) = (p[1] .+ p[2] * x[1,:] + p[3] * x[2,:]) .* (x[2,:].^2)
  # sample = rand(1:length(skeletons), 20)
  # xdata = 
  workArray = [work(s) for s in skeletons]
  perm = sortperm(workArray)
  sums = cumsum(workArray[perm])

  compsize = mean(workArray) * chunksize
  delimiters = ones(Int64, 1)
  k = 1
  while delimiters[end] <= length(workArray) - 1
    ff = findfirst(x -> x > k * compsize, sums)
    if isnothing(ff) || ff >= length(workArray)
      push!(delimiters, length(workArray) + 1)
    else
      push!(delimiters, max(ff, delimiters[k] + 1))
    end
    k += one(k)
  end
  return [skeletons[perm[delimiters[k] : (delimiters[k + 1] - 1)]] for k = 1 : (length(delimiters)-1)]
end