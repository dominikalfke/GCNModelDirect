

export
    FullStandardKernelMatrices,
    ScalingPlusLowRankKernelMatrices



#### Kernels made of simple (dense or sparse) nxn matrices

mutable struct FullStandardKernelMatrices <: StandardKernelMatrices
    kernel :: GCNKernel
    matrices :: Vector # No type to include UniformScaling
    trainingSubmatrices :: Vector

    FullStandardKernelMatrices(kernel :: GCNKernel) =
        new(kernel, [], [])
end

KernelMatrices(k :: FixedMatrixKernel) = FullStandardKernelMatrices(k)
KernelMatrices(k :: PolyLaplacianKernel) = FullStandardKernelMatrices(k)
KernelMatrices(k :: InverseLaplacianKernel) = FullStandardKernelMatrices(k)

function setupMatrices!(:: DirectStandardGCN, km :: FullStandardKernelMatrices, dataset :: Dataset)
    matrices = computeMatrices(km.kernel, dataset)
    if isa(matrices, Matrix{<: Number})
        matrices = [matrices,]
    end
    km.matrices = matrices

    for M in matrices
        if isa(M, UniformScaling)
            s = length(dataset.trainingSet)
            push!(km.trainingSubmatrices,
                sparse(1:s, dataset.trainingSet, fill(M.λ, s), s, dataset.numNodes))
        elseif isa(M, SparseMatrixCSC)
            push!(km.trainingSubmatrices, M[:, dataset.trainingSet]')
        else
            push!(km.trainingSubmatrices, M[dataset.trainingSet, :])
        end
    end
end

applyKernel(km :: FullStandardKernelMatrices, X :: Vector{Matrix{Float64}}) =
    sum(km.matrices[k] * X[k] for k in 1:length(X))

applyKernelBeforeWeights(km :: FullStandardKernelMatrices, X :: AbstractMatrix{Float64}) =
    [M * X for M in km.matrices]

applyKernelTrainingRows(km :: FullStandardKernelMatrices, X :: Vector{Matrix{Float64}}) =
    sum(km.trainingSubmatrices[k] * X[k] for k in 1:length(X))

applyKernelTrainingColumnsBeforeWeights(km :: FullStandardKernelMatrices, X :: AbstractMatrix{Float64}) =
    [M' * X for M in km.trainingSubmatrices]



#### Kernels made up of a scaled identity and a low rank matrix

mutable struct ScalingPlusLowRankKernelMatrices <: StandardKernelMatrices
    kernel :: GCNKernel
    numParts :: Int
    scalingFactors :: Vector{Float64}
    lowRankProjectionMatrix :: Matrix{Float64}
    lowRankInnerMatrices :: Vector # because it might contain UniformScaling
    trainingSubmatrices :: Vector

    ScalingPlusLowRankKernelMatrices(kernel :: GCNKernel) = new(kernel, numParts(kernel),
        Float64[], zeros(0,0), [], [])
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
    set = dataset.trainingSet
    for k in 1:km.numParts
        K = km.lowRankProjectionMatrix[set,:] * km.lowRankInnerMatrices[k] * km.lowRankProjectionMatrix'
        for (i,j) in pairs(set)
            K[i,j] += km.scalingFactors[k]
        end
        push!(km.trainingSubmatrices, K)
    end
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

applyKernelTrainingRows(km :: ScalingPlusLowRankKernelMatrices, X :: Vector{Matrix{Float64}}) =
    sum(km.trainingSubmatrices[k] * X[k] for k in 1:length(X))

applyKernelTrainingColumnsBeforeWeights(km :: ScalingPlusLowRankKernelMatrices, X :: AbstractMatrix{Float64}) =
    [M' * X for M in km.trainingSubmatrices]



#### Kernels made up of a diagonal and a low rank matrix

mutable struct DiagonalPlusLowRankKernelMatrices <: StandardKernelMatrices
    kernel :: GCNKernel
    numParts :: Int
    diagonals :: Vector{Vector{Float64}}
    lowRankProjectionMatrix :: Matrix{Float64}
    lowRankInnerMatrices :: Vector # No {<: AbstractMatrix} because it might contain UniformScaling
    trainingSubmatrices :: Vector

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
    set = dataset.trainingSet
    for k in 1:km.numParts
        K = km.lowRankProjectionMatrix[set,:] * km.lowRankInnerMatrices[k] * km.lowRankProjectionMatrix'
        for (i,j) in pairs(set)
            K[i,j] += km.diagonals[k][j]
        end
        push!(km.trainingSubmatrices, K)
    end
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

applyKernelTrainingRows(km :: DiagonalPlusLowRankKernelMatrices, X :: Vector{Matrix{Float64}}) =
    sum(km.trainingSubmatrices[k] * X[k] for k in 1:length(X))

applyKernelTrainingColumnsBeforeWeights(km :: DiagonalPlusLowRankKernelMatrices, X :: AbstractMatrix{Float64}) =
    [M' * X for M in km.trainingSubmatrices]
