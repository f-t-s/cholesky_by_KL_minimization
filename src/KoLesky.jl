include("SortSparse.jl")
using Distributed
using DistributedArrays
using LinearAlgebra
using IntelVectorMath

#Assumes that the values of y are already written into the buffer.
#They will be overwritten during the execution of the function
function predictSkeleton!( μ, σ, L, B, y, α, β, γ, δ, ℓ, skel, Kij!, ITest )
    NTest = length(ITest)
    NTrain = length(skel.children)
    # We are using reverse ordering during this function
    reverse!(skel.parents)
    reverse!(skel.children)
    reverse!(y)

    Kij!(L, skel.children, skel.children)

    # Compute the cholesky factor of the sub-covariance matrix
    L = cholesky!(Hermitian(L)).L



    # Compute B
    Kij!(B, skel.children, ITest)

    # Compute the intermediate quantities that use inner products for the 
    # First value of k. 
    # They will be updated by substracting the corresponding columns for the different
    # k
    ldiv!(L, B)
    ldiv!(L, y)
    # Should not be necessary but presently mapreduce seems to allocate excessive amounts of memory
    for j = 1 : NTest
        for i = 1 : NTrain
            α[j] += y[i] * B[i,j]
            β[j] += B[i,j]^2
        end
    end

    old_k = length(skel.children) 
    for k_par = length(skel.parents) : -1 : 1 
        k = findlast(x -> x == skel.parents[k_par],
            skel.children)

        for l = (k + 1) : (old_k)
            for j = 1 : NTest
                α[j] -= y[l] * B[l,j] 
                β[j] -= B[l,j]^2
            end
        end
        for j = 1 : NTest
            γ[j] = sqrt(one(eltype(γ)) + (B[k,j])^2 / (δ[j] - β[j]))
        end

        for j = 1 : NTest
            ℓ[j] = - B[k,j] * (one(eltype(ℓ)) + β[j] / (δ[j] - β[j])) / δ[j] / γ[j] 
        end

        for j = 1 : NTest
            μ[j] += ℓ[j] / γ[j] * (y[k] + B[k,j] * α[j] / (δ[j] - β[j]))
        end
        @. σ += ℓ^2
        old_k = k
    end

    # We are reverting the reversion, for now, to not interfere with parts of the
    # code that might require the original ordering.
    reverse!(skel.parents)
    reverse!(skel.children)
end

# Strangely slow
function predict_threaded(skeletons::Vector{Skeleton{Ti}}, ITest::AbstractVector{Ti}, yTrain, Kij!) where {Ti}
    # Computing the required size of the buffers 
    maxChildren = maximum(length.( getfield.(skeletons, :children)))
    maxParents = maximum(length.( getfield.(skeletons, :parents)))
    NTest = length(ITest)
    NTrain = sum( length.(getfield.(skeletons, :parents)))

    
    δ = Vector{Float64}(undef, NTest)

    # Defining Buffers 
    LBuffers =  Vector{Vector{Float64}}(undef, Threads.nthreads())
    BBuffers =  Vector{Vector{Float64}}(undef, Threads.nthreads())
    yBuffers =  Vector{Vector{Float64}}(undef, Threads.nthreads())
    αs =  Vector{Vector{Float64}}(undef, Threads.nthreads())
    βs =  Vector{Vector{Float64}}(undef, Threads.nthreads())
    γs =  Vector{Vector{Float64}}(undef, Threads.nthreads())
    ℓs =  Vector{Vector{Float64}}(undef, Threads.nthreads())
    μs =  Vector{Vector{Float64}}(undef, Threads.nthreads())
    σs =  Vector{Vector{Float64}}(undef, Threads.nthreads())

    # Preallocating buffers 
    for k = 1 : Threads.nthreads()
        LBuffers[k] = Vector{Float64}(undef, maxChildren^2)
        BBuffers[k] = Vector{Float64}(undef, maxChildren * NTest)
        yBuffers[k] = Vector{Float64}(undef, maxChildren)
        αs[k] = Vector{Float64}(undef, NTest)
        βs[k] = Vector{Float64}(undef, NTest)
        γs[k] = Vector{Float64}(undef, NTest)
        ℓs[k] = Vector{Float64}(undef, NTest)
        μs[k] = zeros(NTest)
        σs[k] = zeros(NTest)
    end

    # A little cumbersome since VML.jl doesn't like subarrays
    tmp = Vector{Float64}(undef, 1)
    for i = 1 : NTest
        Kij!(tmp, ITest[i:i], ITest[i:i])
        δ[i] = tmp[1]
    end

    LinearAlgebra.BLAS.set_num_threads(1)

    println("Multithreaded")

    Threads.@threads for k_skel = 1 : length(skeletons)
        skel = skeletons[k_skel]
        k = Threads.threadid()
        nChildren = length(skel.children)
        # Setting up the outputs:
        L = unsafe_wrap(Array{Float64,2}, pointer(LBuffers[k]), (nChildren, nChildren))
        B = unsafe_wrap(Array{Float64,2}, pointer(BBuffers[k]), (nChildren, NTest))
        y = unsafe_wrap(Array{Float64,1}, pointer(yBuffers[k]), nChildren)
        y .= yTrain[skel.children]
        αs[k] .= 0.0
        βs[k] .= 0.0
        γs[k] .= 0.0
        ℓs[k] .= 0.0

        # Do the prediction using the io variables defined above
        predictSkeleton!(μs[k], σs[k], L, B, y, αs[k], βs[k], γs[k], δ, ℓs[k], skel, Kij!, ITest)
    end

    μ = sum(μs)
    σ = sum(σs)

    @. σ .+= 1. / δ
    @. σ = 1. / σ
    @. μ = - σ * μ 
    return μ, σ
end



function predict(skeletons::Vector{Skeleton{Ti}}, ITest::AbstractVector{Ti}, yTrain, Kij!) where {Ti}
    # Computing the required size of the buffers 
    maxChildren = maximum(length.( getfield.(skeletons, :children)))
    maxParents = maximum(length.( getfield.(skeletons, :parents)))
    NTest = length(ITest)
    NTrain = sum( length.(getfield.(skeletons, :parents)))

    μ = zeros(NTest)
    σ = zeros(NTest)
    
    #Preallocating buffers 
    LBuffer = Vector{Float64}(undef, maxChildren^2)
    BBuffer = Vector{Float64}(undef, maxChildren * NTest)
    yBuffer = Vector{Float64}(undef, maxChildren)
    α = Vector{Float64}(undef, NTest)
    β = Vector{Float64}(undef, NTest)
    γ = Vector{Float64}(undef, NTest)
    δ = Vector{Float64}(undef, NTest)
    ℓ = Vector{Float64}(undef, NTest)

    # A little cumbersome since VML.jl doesn't like subarrays
    tmp = Vector{Float64}(undef, 1)
    for i = 1 : NTest
        Kij!(tmp, ITest[i:i], ITest[i:i])
        δ[i] = tmp[1]
    end

    @. σ .= 1. / δ

    for skel in skeletons 
        nChildren = length(skel.children)
        # Setting up the outputs:
        L = unsafe_wrap(Array{Float64,2}, pointer(LBuffer), (nChildren, nChildren))
        B = unsafe_wrap(Array{Float64,2}, pointer(BBuffer), (nChildren, NTest))
        y = unsafe_wrap(Array{Float64,1}, pointer(yBuffer), nChildren)
        y .= yTrain[skel.children]
        α .= 0.0
        β .= 0.0
        γ .= 0.0
        ℓ .= 0.0

        # Do the prediction using the io variables defined above
        predictSkeleton!(μ, σ, L, B, y, α, β, γ, δ, ℓ, skel, Kij!, ITest)
    end

    @. σ = 1. / σ
    @. μ = - σ * μ 
    return μ, σ
end

function predict_distributed( skeletons::Vector{Skeleton{Ti}}, ITest, yTrain, Kij!, chunksize=Ti(20) ) where {Ti}
    NTest = length(ITest)
    # μ = Vector{Float64}(undef, NTest)
    # σ = Vector{Float64}(undef, NTest)
    δ = Vector{Float64}(undef, NTest)

    # A little cumbersome since VML.jl doesn't like subarrays
    tmp = Vector{Float64}(undef, 1)
    for i = 1 : NTest
        Kij!(tmp, ITest[i:i], ITest[i:i])
        δ[i] = tmp[1]
    end
    # σ .= 0

    # @. σ .= 1. / δ

    # @show δ
    # return 0.0

    @everywhere include("./src/KoLesky.jl")
    @everywhere include("./src/SortSparse.jl")

    partitionSkeletons(quadcube, skeletons, chunksize)
    μ::Vector{Float64}, σ::Vector{Float64} = @sync @distributed (+) for skels in distribute(partitionSkeletons(quadcube, skeletons, chunksize))
        maxChildren = maximum(length.( getfield.(skels, :children)))
        maxParents = maximum(length.( getfield.(skels, :parents)))
        NTrain = sum( length.(getfield.(skels, :parents)))

        μLocal = zeros(NTest)
        σLocal = zeros(NTest)

        #Preallocating buffers 
        LBuffer = Vector{Float64}(undef, maxChildren^2)
        BBuffer = Vector{Float64}(undef, maxChildren * NTest)
        yBuffer = Vector{Float64}(undef, maxChildren)
        α = Vector{Float64}(undef, NTest)
        β = Vector{Float64}(undef, NTest)
        γ = Vector{Float64}(undef, NTest)
        ℓ = Vector{Float64}(undef, NTest)

        μloc = zeros(NTest)
        σloc = zeros(NTest)
        for skel in skels
            nChildren = length(skel.children)
            # Setting up the outputs:
            L = unsafe_wrap(Array{Float64,2}, pointer(LBuffer), (nChildren, nChildren))
            B = unsafe_wrap(Array{Float64,2}, pointer(BBuffer), (nChildren, NTest))
            y = unsafe_wrap(Array{Float64,1}, pointer(yBuffer), nChildren)
            y .= yTrain[skel.children]
            α .= 0.0
            β .= 0.0
            γ .= 0.0
            ℓ .= 0.0

            # Do the prediction using the io variables defined above
            predictSkeleton!(μloc, σloc, L, B, y, α, β, γ, δ, ℓ, skel, Kij!, ITest)
        end
        [μloc, σloc]
    end

    @. σ += 1. / δ
    @. σ = 1. / σ
    @. μ = - σ * μ 
    return μ, σ
end


function predict(skeletons::Vector{Skeleton{Ti}}, ITest, yTrain, x::AbstractArray, cov!, σ=zeros(size(x,2))) where {Ti}

    # A function that fills a matrix with the entries of the covariance matrix corresponding to a given set of indices
    function Kij!(Out::AbstractArray, I::AbstractVector, J::AbstractVector)
        @fastmath @inbounds for (indI, i) ∈ enumerate(I)
            for (indJ, j) ∈ enumerate(J)
                Out[indI, indJ] = zero(eltype(x))
                for d = 1 : size(x,1)
                    Out[indI, indJ] += (x[d,i] - x[d,j])^2 
                end
            end
        end
        #compute covariance function
        cov!(IntelVectorMath.sqrt!(Out))

        @fastmath @inbounds for (indI, i) ∈ enumerate(I)
            for (indJ, j) ∈ enumerate(J)
                if i == j
                    Out[indI, indJ] += σ[i]
                end
            end
        end
    end

    return μ, σ = predict_distributed(skeletons, ITest, yTrain, Kij!, 100)
    # return μ, σ = predict(skeletons, ITest, yTrain, Kij!)
end

# a function that generates a sparse matrix with the sparsity pattern corresponding to a vector of skeletons
function assembleL(skeletons::Vector{Skeleton{Ti}}) where {Ti}
    IJ = [ij for s in skeletons for ij in (Iterators.product(s.children, s.parents))]
    IJ = filter(ij -> ij[1] >= ij[2], IJ) 
    I = getindex.(IJ,1)
    J = getindex.(IJ,2)
    return(sparse(I, J, zeros(length(I))))
end

function assembleLSkeleton!(U, B, skel, Kij!)
    # We are using reverse ordering during this function
    reverse!(skel.parents)
    reverse!(skel.children)
    Kij!(U, skel.children, skel.children)
    # Compute the cholesky factor of the sub-covariance matrix

    U = cholesky!(Hermitian(U)).U

    # We are reverting the reversion, for now, to not interfere with parts of the
    # code that might require the original ordering.

    
    # Computes the leading k columns of the inverse of L
    B .= 0.0 
    for (i, par) in enumerate(skel.parents)
        B[findlast(x -> x == par, skel.children), i] = 1.0
    end
    # B .= U \ B
    ldiv!(U, B)

    reverse!(vec(B))
    reverse!(skel.parents)
    reverse!(skel.children)
end


function assembleL(skeletons::Vector{Skeleton{Ti}}, x::AbstractArray, cov!, σ=zeros(size(x, 2))) where {Ti}
    # A function that fills a matrix with the entries of the covariance matrix corresponding to a given set of indices
    function Kij!(Out::AbstractArray, I::AbstractVector, J::AbstractVector)
        @fastmath @inbounds for (indI, i) ∈ enumerate(I)
            for (indJ, j) ∈ enumerate(J)
                Out[indI, indJ] = zero(eltype(x))
                for d = 1 : size(x,1)
                    Out[indI, indJ] += (x[d,i] - x[d,j])^2 
                end
            end
        end

        #compute covariance function
        cov!(IntelVectorMath.sqrt!(Out))

        @fastmath @inbounds for (indI, i) ∈ enumerate(I)
            for (indJ, j) ∈ enumerate(J)
                if i == j
                    Out[indI, indJ] += σ[i]
                end
            end
        end
    end

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
    LBuffer = Vector{Float64}(undef, maxChildren^2)
    BBuffer = Vector{Float64}(undef, maxChildren * maxParents)


    for skel in skeletons
        nChildren = length(skel.children)
        nParents = length(skel.parents)
        # Setting up the outputs:
        L = unsafe_wrap(Array{Float64,2}, pointer(LBuffer), (nChildren, nChildren))
        B = unsafe_wrap(Array{Float64,2}, pointer(BBuffer), (nChildren, nParents))

            
        # Do the prediction using the io variables defined above
        assembleLSkeleton!(L, B, skel, Kij!)
        for (l, tup) in enumerate(Iterators.product(skel.children, skel.parents))
            I[offset + l] = tup[1]
            J[offset + l] = tup[2]
            S[offset + l] = B[l] 
        end
        offset += nChildren * nParents
    end

    # Selecting the lower triangular part
    inds = findall(I .>= J)
    I, J, S = I[inds], J[inds], S[inds]

    # Construct sparse matrix and return
    return sparse(I,J,S)
end

