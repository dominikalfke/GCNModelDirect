

export
    LowRankKernelMatrices,
    computeKernelFactorization,
    DirectLowRankGCN


"""
    LowRankKernelMatrices

`KernelMatrices` subtype for all kernel types whose kernel matrices have a
low-rank factorization with a shared reduction matrix. For custom `GCNKernel`
subtypes satisfying this condition, simply add a function
`KernelMatrices(k :: CustomKernel) = LowRankKernelmatrices(k, rank(k))`
(where `rank(k)` is replaced with a suitable expression) as well as an
implementation of `computeKernelFactorization`.
"""
struct LowRankKernelMatrices <: KernelMatrices
    kernel :: GCNKernel
    rank :: Int
end

"""
    computeKernelFactorization(kernel :: GCNKernel, dataset :: Dataset)

For low-rank `GCNKernel` subtypes, compute the factorization of the kernel
matrices. The returned value has to be a tuple `(U, d)` where `U` is the shared
`n × r` reduction matrix, and `diagonals` is a 1-D array holding the `K` vectors
on the diagonals of the inner parts, i.e., the filter function values. Here `K`
is the number of kernel parts. Note that even if `K==1`, the returned value must
be a vector holding a single value, which is the diagonal vector.
"""
computeKernelFactorization(kernel :: GCNKernel, dataset :: Dataset) =
    error("No kernel factorization available for a kernel of type $(typeof(kernel))")

DirectGCN(arc :: GCNArchitecture, dataset :: Dataset, kernel :: LowRankKernelMatrices,
            activation :: ActivationMatrices; kwargs...) =
    DirectLowRankGCN(arc, dataset, kernel.rank, activation; kwargs...)


"""
    DirectLowRankGCN

Subtype of `DirectGCN` for implementation of a GCN that exploits kernels where
each part has a low-rank factorization with a shared reduction matrix. Unlike
`DirectStandardGCN`, this fact can be exploited to
"""
mutable struct DirectLowRankGCN <: DirectGCN

    architecture :: GCNArchitecture
    dataset :: Dataset
    rank :: Int
    activation :: ActivationMatrices

    numLayers :: Int
    numKernelParts :: Int

    reductionMatrix :: Matrix{Float64}
    kernelDiagonals :: Vector{Vector{Float64}}
    weightMatrices :: Matrix{Matrix{Float64}}

    reducedInputAfterKernels :: Vector{Matrix{Float64}}
    reducedHiddenLayersBeforeActivation :: Vector{Matrix{Float64}}
    reducedHiddenLayers :: Vector{Matrix{Float64}}
    reducedOutput :: Matrix{Float64}

    trainingReductionSubmatrix :: Matrix{Float64}
    trainingLabels :: Matrix{Float64}
    optimizerState

    function DirectLowRankGCN(arc :: GCNArchitecture, dataset :: Dataset,
                rank :: Int, activation :: ActivationMatrices)

        self = new(arc, dataset, rank, activation)

        checkCompatibility(arc, dataset)

        self.numLayers = length(arc.layerWidths)-1
        self.numKernelParts = numParts(arc.kernel)

        self.reductionMatrix, self.kernelDiagonals = computeKernelFactorization(arc.kernel, dataset)
        setupMatrices!(self, self.activation, dataset)

        self.weightMatrices = Matrix{Matrix{Float64}}(undef, self.numLayers, self.numKernelParts)
        initializeRandomWeights!(self)

        reducedInput = self.reductionMatrix' * dataset.features
        self.reducedInputAfterKernels = [φ .* reducedInput for φ in self.kernelDiagonals]
        self.reducedHiddenLayersBeforeActivation = Vector{Matrix{Float64}}(undef, self.numLayers-1)
        self.reducedHiddenLayers = Vector{Matrix{Float64}}(undef, self.numLayers-1)
        propagateLayers!(self)

        self.trainingReductionSubmatrix = self.reductionMatrix[dataset.trainingSet, :]
        self.trainingLabels = dataset.labels[dataset.trainingSet, :]
        self.optimizerState = nothing

        return self
    end
end

Base.show(io :: IO, gcn :: DirectLowRankGCN) = print(io,
    "$(typeof(gcn))($(gcn.architecture), $(gcn.dataset))")

function initializeRandomWeights!(gcn :: DirectLowRankGCN)
    for l = 1:gcn.numLayers
        inputWidth = gcn.architecture.layerWidths[l]
        outputWidth = gcn.architecture.layerWidths[l+1]
        initRange = sqrt(6.0/(inputWidth+outputWidth))
        for k = 1:gcn.numKernelParts
            gcn.weightMatrices[l,k] = initRange *
                (1 .- 2*rand(Float64, inputWidth, outputWidth))
        end
    end
end

function propagateLayers!(gcn :: DirectLowRankGCN)
    X = sum(gcn.reducedInputAfterKernels[k] * gcn.weightMatrices[1,k]
                for k = 1:gcn.numKernelParts)
    for l = 1:gcn.numLayers-1
        gcn.reducedHiddenLayersBeforeActivation[l] = X
        X = applyActivation(gcn, gcn.activation, X)
        gcn.reducedHiddenLayers[l] = X
        X = sum(gcn.kernelDiagonals[k] .* (X * gcn.weightMatrices[l+1,k])
                for k = 1:gcn.numKernelParts)
    end
    gcn.reducedOutput = X
end

output(gcn :: DirectLowRankGCN) =
    gcn.reductionMatrix * gcn.reducedOutput

trainingOutput(gcn :: DirectLowRankGCN) =
    gcn.trainingReductionSubmatrix * gcn.reducedOutput

function computeParameterGradients(gcn :: DirectLowRankGCN)
    gradients = Matrix{Matrix{Float64}}(undef, gcn.numLayers, gcn.numKernelParts)

    dX = (trainingClassProbabilities(gcn) - gcn.trainingLabels) / length(gcn.dataset.trainingSet)
    dX = gcn.trainingReductionSubmatrix' * dX

    for l = gcn.numLayers:-1:1
        if l < gcn.numLayers
            dX = backpropagateActivationDerivative(gcn, gcn.activation,
                    gcn.reducedHiddenLayersBeforeActivation[l], dX)
        end
        if l == 1
            for k = 1:gcn.numKernelParts
                gradients[1,k] = gcn.reducedInputAfterKernels[k]' * dX +
                            gcn.architecture.regParam * gcn.weightMatrices[l,k]
            end
        else
            for k = 1:gcn.numKernelParts
                gradients[l,k] = gcn.reducedHiddenLayers[l-1]' * (gcn.kernelDiagonals[k] .* dX)
            end
            dX = sum((gcn.kernelDiagonals[k] .* dX) * gcn.weightMatrices[l,k]'
                        for k in 1:gcn.numKernelParts)
        end
    end
    return gradients
end


function updateParameters!(gcn :: DirectLowRankGCN, Θ :: Matrix{Matrix{Float64}}, factor :: Float64 = 1.0)
    for (index, T) in pairs(Θ)
        axpy!(factor, T, gcn.weightMatrices[index])
    end
    propagateLayers!(gcn)
end
