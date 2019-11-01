

export
    FullStandardKernelMatrices,
    ScalingPlusLowRankKernelMatrices



### IdentityKernel

struct IdentityKernelMatrices <: StandardKernelMatrices end
KernelMatrices(:: IdentityKernel) = IdentityKernelMatrices()

applyKernel(::IdentityKernelMatrices, X :: Vector{Matrix{Float64}}) =
    X[1]

applyKernelBeforeWeights(::IdentityKernelMatrices, X :: AbstractMatrix{Float64}) =
    [X,]

applyKernelRows(::IdentityKernelMatrices, X :: Vector{Matrix{Float64}}, indexSet) =
    X[1][indexSet,:]

applyKernelColumnsBeforeWeights(::IdentityKernelMatrices, X :: AbstractMatrix{Float64}, indexSet) =
    [X[indexSet,:],]


#### Kernels made of simple (dense or sparse) nxn matrices

mutable struct FullStandardKernelMatrices <: StandardKernelMatrices
    kernel :: GCNKernel
    matrices :: Vector{Any} # Any to include UniformScaling
    FullStandardKernelMatrices(kernel :: GCNKernel) =
        new(kernel, [])
end

KernelMatrices(k :: FixedMatrixKernel) = FullStandardKernelMatrices(k)
KernelMatrices(k :: PolyLaplacianKernel) = FullStandardKernelMatrices(k)
KernelMatrices(k :: InverseLaplacianKernel) = FullStandardKernelMatrices(k)

function setupMatrices!(:: DirectStandardGCN, km :: FullStandardKernelMatrices, dataset :: Dataset)
    mat = computeMatrices(km.kernel, dataset)
    if isa(mat, Matrix{<: Number})
        mat = [mat,]
    end
    km.matrices = mat
end

applyKernel(km :: FullStandardKernelMatrices, X :: Vector{Matrix{Float64}}) =
    sum(km.matrices[k] * X[k] for k in 1:length(X))

applyKernelBeforeWeights(km :: FullStandardKernelMatrices, X :: AbstractMatrix{Float64}) =
    [M * X for M in km.matrices]

applyKernelRows(km :: FullStandardKernelMatrices, X :: Vector{Matrix{Float64}}, indexSet) =
    sum(km.matrices[k][indexSet,:] * X[k] for k in 1:length(X))

applyKernelColumnsBeforeWeights(km :: FullStandardKernelMatrices, X :: AbstractMatrix{Float64}, indexSet) =
    [M[:,indexSet] * X for M in km.matrices]



#### Kernels made up of a scaled identity and a low rank matrix

mutable struct ScalingPlusLowRankKernelMatrices <: StandardKernelMatrices
    kernel :: GCNKernel
    numParts :: Int
    scalingFactors :: Vector{Float64}
    lowRankProjectionMatrix :: Matrix{Float64}
    lowRankInnerMatrices :: Vector{<: Any} # because it might contain UniformScaling

    ScalingPlusLowRankKernelMatrices(kernel :: GCNKernel) = new(kernel, numParts(kernel),
        Float64[], zeros(0,0), AbstractMatrix{Float64}[])
end

KernelMatrices(kernel :: PolyHypergraphLaplacianKernel) =
    ScalingPlusLowRankKernelMatrices(kernel)
computeScalingPlusLowRankMatrixParts(kernel :: PolyHypergraphLaplacianKernel, dataset :: Dataset) =
    computeMatrices(kernel, dataset)

KernelMatrices(kernel :: InvHypergraphLaplacianKernel) =
    ScalingPlusLowRankKernelMatrices(kernel)
function computeScalingPlusLowRankMatrixParts(kernel :: InvHypergraphLaplacianKernel, dataset :: Dataset)
    λmin, U, innerDiag = computeMatrices(kernel, dataset)
    return [λmin], U, [Diagonal(innerDiag)]
end

function setupMatrices!(:: DirectStandardGCN, km :: ScalingPlusLowRankKernelMatrices, dataset :: Dataset)
    km.scalingFactors, km.lowRankProjectionMatrix, km.lowRankInnerMatrices =
        computeScalingPlusLowRankMatrixParts(km.kernel, dataset)
end



function applyKernel(km :: ScalingPlusLowRankKernelMatrices, X :: Vector{Matrix{Float64}})
    Y = km.lowRankProjectionMatrix * sum(km.lowRankInnerMatrices[k] *
                        (km.lowRankProjectionMatrix' * X[k]) for k in km.numParts)
    for k in 1:km.numParts
        axpy!(km.scalingFactors[k], X[k], Y)
    end
    return Y
end

function applyKernelBeforeWeights(km :: ScalingPlusLowRankKernelMatrices, X :: AbstractMatrix{Float64})
    projectedX = km.lowRankProjectionMatrix' * X
    return [km.scalingFactors[k]*X + km.lowRankProjectionMatrix * (km.lowRankInnerMatrices[k] * projectedX)
            for k in 1:km.numParts]
end

function applyKernelRows(km :: ScalingPlusLowRankKernelMatrices, X :: Vector{Matrix{Float64}}, indexSet)
    Y = km.lowRankProjectionMatrix[indexSet,:] *
            sum(km.lowRankInnerMatrices[k] * (km.lowRankProjectionMatrix' * X[k]) for k in km.numParts)
    for k in 1:km.numParts
        axpy!(km.scalingFactors[k], X[k][indexSet,:], Y)
    end
    return Y
end

function applyKernelColumnsBeforeWeights(km :: ScalingPlusLowRankKernelMatrices, X :: AbstractMatrix{Float64}, indexSet)
    projectedX = km.lowRankProjectionMatrix[indexSet, :]' * X
    Y = Vector{Matrix{Float64}}(undef, km.numParts)
    for k in 1:km.numParts
        Y[k] = km.lowRankProjectionMatrix * (km.lowRankInnerMatrices[k] * projectedX)
        Y[k][indexSet, :] += km.scalingFactors[k]*X
    end
    return Y
end



#### Kernels made up of a diagonal and a low rank matrix

mutable struct DiagonalPlusLowRankKernelMatrices <: StandardKernelMatrices
    kernel :: GCNKernel
    numParts :: Int
    diagonals :: Vector{Vector{Float64}}
    lowRankProjectionMatrix :: Matrix{Float64}
    lowRankInnerMatrices :: Vector{<: Any} # No {<: AbstractMatrix} because it might contain UniformScaling

    DiagonalPlusLowRankKernelMatrices(kernel :: GCNKernel) = new(kernel, numParts(kernel),
        Vector{Float64}[], zeros(0,0), AbstractMatrix{Float64}[])
end

KernelMatrices(kernel :: PolySmoothedHypergraphLaplacianKernel) =
    DiagonalPlusLowRankKernelMatrices(kernel)
computeDiagonalPlusLowRankMatrixParts(kernel :: PolySmoothedHypergraphLaplacianKernel, dataset :: Dataset) =
    computeMatrices(kernel, dataset)

function setupMatrices!(:: DirectStandardGCN, km :: DiagonalPlusLowRankKernelMatrices, dataset :: Dataset)
    km.diagonals, km.lowRankProjectionMatrix, km.lowRankInnerMatrices =
        computeDiagonalPlusLowRankMatrixParts(km.kernel, dataset)
end



function applyKernel(km :: DiagonalPlusLowRankKernelMatrices, X :: Vector{Matrix{Float64}})
    Y = km.lowRankProjectionMatrix * sum(km.lowRankInnerMatrices[k] *
                        (km.lowRankProjectionMatrix' * X[k]) for k in km.numParts)
    for k in 1:km.numParts
        axpy!(1.0, km.diagonals[k] .* X[k], Y)
    end
    return Y
end

function applyKernelBeforeWeights(km :: DiagonalPlusLowRankKernelMatrices, X :: AbstractMatrix{Float64})
    projectedX = km.lowRankProjectionMatrix' * X
    return [km.diagonals[k] .* X + km.lowRankProjectionMatrix * (km.lowRankInnerMatrices[k] * projectedX)
            for k in 1:km.numParts]
end

function applyKernelRows(km :: DiagonalPlusLowRankKernelMatrices, X :: Vector{Matrix{Float64}}, indexSet)
    Y = km.lowRankProjectionMatrix[indexSet,:] *
            sum(km.lowRankInnerMatrices[k] * (km.lowRankProjectionMatrix' * X[k]) for k in km.numParts)
    for k in 1:km.numParts
        axpy!(1.0, km.diagonals[k][indexSet] .* X[k][indexSet,:], Y)
    end
    return Y
end

function applyKernelColumnsBeforeWeights(km :: DiagonalPlusLowRankKernelMatrices, X :: AbstractMatrix{Float64}, indexSet)
    projectedX = km.lowRankProjectionMatrix[indexSet, :]' * X
    Y = Vector{Matrix{Float64}}(undef, km.numParts)
    for k in 1:km.numParts
        Y[k] = km.lowRankProjectionMatrix * (km.lowRankInnerMatrices[k] * projectedX)
        Y[k][indexSet, :] += km.diagonals[k][indexSet] .* X
    end
    return Y
end
