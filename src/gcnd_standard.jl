
export
    StandardKernelMatrices,
    applyKernel,
    applyKernelBeforeWeights,
    applyKernelTrainingRows,
    applyKernelTrainingColumnsBeforeWeights,
    DirectStandardGCN

"""
    StandardKernelMatrices

Abstract intermediate type for `KernelMatrices` subtypes that warrant a standard
(non-lowrank) implementation. All subtypes must implement the four functions
`applyKernel`, `applyKernelBeforeWeights`, `applyKernelTrainingRows`, and
`applyKernelTrainingColumnsBeforeWeights`
"""
abstract type StandardKernelMatrices <: KernelMatrices end


"""
    applyKernel(km :: StandardKernelMatrices, X :: Vector{Matrix{Float64}})

For a 1-D array of matrices X, compute the sum over all `K_k*X[k]`, where
`K_k` is the k-th kernel matrix.
"""
applyKernel(:: StandardKernelMatrices, X :: Vector{Matrix{Float64}}) = nothing

"""
    applyKernelBeforeWeights(km :: StandardKernelMatrices, X :: AbstractMatrix{Float64})

For a single abstract matrix X, compute the 1-D array holding all `K_k*X`,
where `K[k]` is the k-th kernel matrix.
"""
applyKernelBeforeWeights(:: StandardKernelMatrices, X :: AbstractMatrix{Float64}) = nothing

"""
    applyKernelTrainingRows(km :: StandardKernelMatrices, X :: Vector{Matrix{Float64}})

For a 1-D array of matrices X, compute the sum over all `K_k[trainingSet,:]*X[k]`, where
`K_k` is the k-th kernel matrix.
"""
applyKernelTrainingRows(:: StandardKernelMatrices, X :: Vector{Matrix{Float64}}) = nothing

"""
    applyKernelTrainingColumnsBeforeWeights(km :: StandardKernelMatrices, X :: AbstractMatrix{Float64})

For a single matrix X, compute the 1-D array holding all `K_k[:,trainingSet]*X`,
where `K_k` is the k-th kernel matrix.
"""
applyKernelTrainingColumnsBeforeWeights(:: StandardKernelMatrices, X :: AbstractMatrix{Float64}) = nothing


function DirectGCN(arc :: GCNArchitecture, dataset :: Dataset,
            kernel :: StandardKernelMatrices, activation :: ActivationMatrices)
    return DirectStandardGCN(arc, dataset, kernel, activation)
end


"""
    DirectStandardGCN

`DirectGCN` subtype that gives an efficient GCN implementation for most kernels.
While kernel matrix structure can be exploited via the `KernelMatrices` subtype,
that structure will not be exploited any further in the GCN.
"""
mutable struct DirectStandardGCN <: DirectGCN

    architecture :: GCNArchitecture
    dataset :: Dataset
    kernel :: StandardKernelMatrices
    activation :: ActivationMatrices

    numLayers :: Int
    numKernelParts :: Int

    weightMatrices :: Matrix{Matrix{Float64}}
    inputAfterKernels :: Vector{Matrix{Float64}}
    hiddenLayersBeforeActivation :: Vector{Matrix{Float64}}
    hiddenLayers :: Vector{Matrix{Float64}}
    outputBeforeKernels :: Vector{Matrix{Float64}}

    trainingLabels :: AbstractMatrix{Float64}
    optimizerState

    function DirectStandardGCN(arc :: GCNArchitecture, dataset :: Dataset,
                    kernel :: StandardKernelMatrices, activation :: ActivationMatrices)

        self = new(arc, dataset, kernel, activation)

        self.numLayers = length(arc.layerWidths)-1
        self.numKernelParts = numParts(arc.kernel)

        setupMatrices!(self, kernel, dataset)
        setupMatrices!(self, activation, dataset)
        self.weightMatrices = Matrix{Matrix{Float64}}(undef, self.numLayers, self.numKernelParts)
        initializeRandomWeights!(self)

        self.inputAfterKernels = applyKernelBeforeWeights(kernel, dataset.features)
        self.hiddenLayersBeforeActivation = Vector{Matrix{Float64}}(undef, self.numLayers-1)
        self.hiddenLayers = Vector{Matrix{Float64}}(undef, self.numLayers-1)
        propagateLayers!(self)

        self.trainingLabels = dataset.labels[dataset.trainingSet, :]
        self.optimizerState = nothing

        return self
    end
end

Base.show(io :: IO, gcn :: DirectStandardGCN) = print(io,
    "$(typeof(gcn))($(gcn.architecture), $(gcn.dataset))")


function initializeRandomWeights!(gcn :: DirectStandardGCN)
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


function propagateLayers!(gcn :: DirectStandardGCN)
    X = sum(gcn.inputAfterKernels[k] * gcn.weightMatrices[1,k]
                for k in 1:gcn.numKernelParts)
    for l = 1:gcn.numLayers-1
        gcn.hiddenLayersBeforeActivation[l] = X
        X = applyActivation(gcn, gcn.activation, X)
        gcn.hiddenLayers[l] = X
        X = [X * Θ for Θ in gcn.weightMatrices[l+1,:]]
        if l+1 < gcn.numLayers
            X = applyKernel(gcn.kernel, X)
        end
    end
    gcn.outputBeforeKernels = X
end

output(gcn :: DirectStandardGCN) =
    applyKernel(gcn.kernel, gcn.outputBeforeKernels)

trainingOutput(gcn :: DirectStandardGCN) =
    applyKernelTrainingRows(gcn.kernel, gcn.outputBeforeKernels)


function computeParameterGradients(gcn :: DirectStandardGCN)
    gradients = Matrix{Matrix{Float64}}(undef, gcn.numLayers, gcn.numKernelParts)

    dX = (trainingClassProbabilities(gcn) - gcn.trainingLabels) / length(gcn.dataset.trainingSet)

    KdX = applyKernelTrainingColumnsBeforeWeights(gcn.kernel, dX)
    for k in 1:gcn.numKernelParts
        gradients[end,k] = gcn.hiddenLayers[end]' * KdX[k]
    end

    for l = gcn.numLayers-1:-1:1
        dX = sum(KdX[k] * gcn.weightMatrices[l+1,k]' for k in 1:gcn.numKernelParts)
        dX = backpropagateActivationDerivative(gcn, gcn.activation,
                    gcn.hiddenLayersBeforeActivation[l], dX)
        if l == 1
            for k in 1:gcn.numKernelParts
                gradients[1,k] = gcn.inputAfterKernels[k]' * dX +
                                    gcn.architecture.regParam * gcn.weightMatrices[1,k]
            end
        else
            KdX = applyKernelBeforeWeights(gcn.kernel, dX)
            for k in 1:gcn.numKernelParts
                gradients[l,k] = gcn.hiddenLayers[l-1]' * KdX[k]
            end
        end
    end
    return gradients
end



function updateParameters!(gcn :: DirectStandardGCN, dΘ :: Matrix{Matrix{Float64}}, factor :: Float64 = 1.0)
    for (index, dW) in pairs(dΘ)
        axpy!(factor, dW, gcn.weightMatrices[index])
    end
    propagateLayers!(gcn)
end
