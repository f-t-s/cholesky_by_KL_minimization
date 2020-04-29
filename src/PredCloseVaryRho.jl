include("SortSparse.jl")
include("KoLesky.jl")
include("Cholesky.jl")
include("CovFuncs.jl")
include("Utils.jl")
using LinearAlgebra
using Base.Threads
using Distributions
using Distances
using Random
using Match
using JLD
using BlockArrays
using ArgParse
Random.seed!(123)

s = ArgParseSettings()
@add_arg_table s begin
  "maternOrder"
  help = "The order of the matern kernel to be used. Should be in (\"12\", \"32\", \"52\")"
  required = true
end
maternOrder = parse_args(s)["maternOrder"]

NTrain = 5000
NTest = 100
NSamples = 10
N = NTrain + NTest
d = 2
l = 0.5
λ = 1.0
ρList = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

meanRMSEPredFirst = zeros(length(ρList))
stdRMSEPredFirst = zeros(length(ρList))
meanRMSEPredLast = zeros(length(ρList))
stdRMSEPredLast = zeros(length(ρList))
meanRMSENoPred = zeros(length(ρList))
stdRMSENoPred = zeros(length(ρList))

# Creating uniform random data
xTrain = rand(d, NTrain)
xTest = rand(d, NTest)
x = hcat(xTrain, xTest)

covfunc12!(r) = matern12!(r,l)
covfunc32!(r) = matern32!(r,l)
covfunc52!(r) = matern52!(r,l)


@match maternOrder begin
  "12" => (covfunc! = covfunc12!)
  "32" => (covfunc! = covfunc32!)
  "52" => (covfunc! = covfunc52!)
end

for (ρInd, ρ)  in enumerate(ρList)

  global meanRMSEPredFirst
  global stdRMSEPredFirst
  global meanRMSEPredLast
  global stdRMSEPredLast
  global meanRMSENoPred
  global stdRMSENoPred
    
    for k = 1 : NSamples

    # =====================================
    # Now construct the proper covariance matrix
    # =====================================

    Θ = covfunc!(pairwise(Euclidean(), x, dims=2))
    chol_Θ = cholesky(Matrix(Θ))

    Θ = PseudoBlockArray(Θ, [NTrain, NTest], [NTrain, NTest])
    chol_Θ_train = cholesky(Θ[Block(1,1)])
    # =====================================
    # Starting with interpolation:
    # =====================================
    
    # Creating reference solution 
    y = chol_Θ.L * randn(N)
    
    # Creating data
    y = PseudoBlockArray(y, [NTrain, NTest])
    
    # Computing the true posterior mean
    μ = Θ[Block(2,1)] * (chol_Θ_train \ y[Block(1)])

    # only needs to be done for one sample 
    Σ = Θ[Block(2,2)] - Θ[Block(2,1)] * (chol_Θ_train \ Θ[Block(1,2)])
    
    # =====================================
    # Computing posterior mean for prediction variables last
    # =====================================
    
    # reordering points and forming skeletons
    colptr, rowval, P, revP, distances = sortSparseRev(xTrain, xTest, ρ, 1)
    xOrd = x[:, P]
    xOrdTest = xOrd[:, 1 : NTest]
    xOrdTrain = xOrd[:, (NTest + 1) : end]
    skeletons = construct_skeletons(colptr, rowval, distances, λ)
    μOrd = μ[P[1 : NTest] .- NTrain]
    ΘOrd = Θ[P, P]
    ΣOrd = Σ[P[1 : NTest] .- NTrain, P[1 : NTest] .- NTrain]
    yOrd = PseudoBlockArray(y[P], [NTest, NTrain]) 

    # Creating vecchia approximation
    L = assembleL(skeletons, xOrd, covfunc!)
    L22 = L[(NTest + 1) : end, (NTest + 1) : end]
    μPredLast = (L' \ ( L \ vcat(zero(y[Block(2)]), L22 * (L22' * (yOrd[Block(2)])))))[1:NTest]
    
    ΣPredLast = inv(Matrix(L[1 : NTest, 1 : NTest] * L[1 : NTest, 1 : NTest]'))

    ΘPredLast = inv(Matrix(L * L'))
    
    #added rescaling with std
    meanRMSEPredLast[ρInd] += sum((μOrd - μPredLast).^2 ./ diag(ΣOrd))

    # only needs to be done once
    stdRMSEPredLast[ρInd] = sqrt(mean((sqrt.(diag(ΣOrd)) - sqrt.(diag(ΣPredLast))).^2 / diag(ΣOrd)))

    # =====================================
    # Computing posterior mean for prediction variables first
    # =====================================
    
    # reordering points and forming skeletons
    colptr, rowval, P, revP, distances = sortSparseRev(xTrain, ρ, 1)
    xOrdTrain = xTrain[:, P]
    xOrd = hcat(xOrdTrain, xTest)
    skeletons = construct_skeletons(colptr, rowval, distances, λ)
    μOrd = μ
    ΘOrd = Θ[vcat(P, (NTrain + 1) : N), vcat(P, (NTrain + 1) : N)]
    ΣOrd = Σ
    yOrd = y[Block(1)][P]

    μPredFirst, ΣPredFirst = predict(skeletons, (NTrain+1): N, yOrd, xOrd, covfunc!)
    
    # added scaling by std
    meanRMSEPredFirst[ρInd] += sum((μOrd - μPredFirst).^2 ./ diag(ΣOrd))
    stdRMSEPredFirst[ρInd] = sqrt(mean((sqrt.(diag(ΣOrd)) - sqrt.(ΣPredFirst)).^2 ./ diag(ΣOrd)))
    
    # =====================================
    # Don't include prediction variables at all
    # =====================================
    
    # reordering points and forming skeletons
    colptr, rowval, P, revP, distances = sortSparseRev(xTrain, ρ, 1)
    xOrdTrain = xTrain[:, P]
    xOrd = hcat(xOrdTrain, xTest)
    skeletons = construct_skeletons(colptr, rowval, distances, λ)
    L = assembleL(skeletons, xOrdTrain, covfunc!)
    @show nnz(L) * 2 / NTrain / (NTrain + 1)
    μOrd = μ
    ΘOrd = Θ[vcat(P, (NTrain + 1) : N), vcat(P, (NTrain + 1) : N)]
    ΣOrd = Σ
    yOrd = y[Block(1)][P]
    
    TMat = pairwise(Euclidean(), xOrdTrain, xTest, dims=2)
    covfunc!(TMat)
    CMat = pairwise(Euclidean(), xTest, dims=2)
    covfunc!(CMat)

    ΣNoPred = CMat - TMat' * L * L' * TMat

    ΘNoPred = inv(Matrix(L * L'))
    ΘNoPred = vcat(ΘNoPred, TMat')
    ΘNoPred = hcat(ΘNoPred, vcat(TMat, CMat))

    stdRMSENoPred[ρInd] = sqrt.(mean((sqrt.(diag(ΣOrd)) - sqrt.(max.(diag(ΣNoPred), 0.0))).^2 ./ diag(ΣOrd)))

    μNoPred = vec(yOrd' * L * L' * TMat)
    meanRMSENoPred[ρInd] += sum((μOrd - μNoPred).^2 ./ diag(ΣOrd))
  end 
end
meanRMSENoPred /= NSamples * NTest
meanRMSENoPred .= sqrt.(meanRMSENoPred)
meanRMSEPredFirst /= NSamples * NTest
meanRMSEPredFirst .= sqrt.(meanRMSEPredFirst)
meanRMSEPredLast /= NSamples * NTest
meanRMSEPredLast .= sqrt.(meanRMSEPredLast)

save("./out/jld/IncludePredCloseVaryRho$maternOrder.jld", 
     "meanRMSEPredFirst", meanRMSEPredFirst,
     "stdRMSEPredFirst", stdRMSEPredFirst,
     "meanRMSEPredLast", meanRMSEPredLast,
     "stdRMSEPredLast", stdRMSEPredLast,
     "meanRMSENoPred", meanRMSENoPred,
     "stdRMSENoPred", stdRMSENoPred,
     "xTrain", xTrain,
     "xTest", xTest, 
     "ρList", ρList)