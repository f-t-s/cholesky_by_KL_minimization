using LinearAlgebra

include("./BEMUtils.jl")



hwlist = haarCube(2)
chargeList = points(1.0 .+ rand(3, 100))

# BEMatrix = assembleMatrix(Φ, hwlist, hwlist)

@time constructMatrixElement(Φ, hwlist[1], hwlist[3])

