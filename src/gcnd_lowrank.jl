

export
    LowRankKernelMatrices,
    DirectLowRankGCN,
    initializeRandomWeights!,
    propagateLayers!,
    output,
    gradientDescentStep!


struct LowRankKernelMatrices <: KernelMatrices
    kernel :: GCNKernel
    rank :: Int
end

DirectGCN(arc :: GCNArchitecture, dataset :: Dataset, kernel :: LowRankKernelMatrices,
            activation :: ActivationMatrices; kwargs...) =
    DirectLowRankGCN(arc, dataset, kernel.rank, activation; kwargs...)

mutable struct DirectLowRankGCN <: DirectGCN

    architecture :: GCNArchitecture
    dataset :: Dataset
    kernel :: GCNKernel
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

    function DirectLowRankGCN(arc :: GCNArchitecture, dataset :: Dataset,
                rank :: Int, activation :: ActivationMatrices)

        self = new(arc, dataset, arc.kernel, rank, activation)

        checkCompatibility(arc, dataset)

        self.numLayers = length(arc.layerWidths)-1
        self.numKernelParts = numParts(arc.kernel)

        self.reductionMatrix, self.kernelDiagonals = computeKernelFactorization(self.kernel, dataset)
        setupMatrices!(self, self.activation, dataset)

        self.weightMatrices = Matrix{Matrix{Float64}}(undef, self.numLayers, self.numKernelParts)
        initializeRandomWeights!(self)

        reducedInput = self.reductionMatrix' * dataset.features
        self.reducedInputAfterKernels = [φ .* reducedInput for φ in self.kernelDiagonals]
        self.reducedHiddenLayersBeforeActivation = Vector{Matrix{Float64}}(undef, self.numLayers-1)
        self.reducedHiddenLayers = Vector{Matrix{Float64}}(undef, self.numLayers-1)
        propagateLayers!(self)

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

output(gcn :: DirectLowRankGCN, index :: Int) =
    gcn.reducedOutput' * gcn.reductionMatrix[index, :]
output(gcn :: DirectLowRankGCN, set = 1:gcn.dataset.numNodes) =
    gcn.reductionMatrix[set, :] * gcn.reducedOutput


function gradientDescentStep!(gcn :: DirectLowRankGCN; stepLength :: Float64 = 0.2)
    trainingSet = gcn.dataset.trainingSet
    dX = gcn.reductionMatrix[trainingSet, :]' *
        (classProbabilities(gcn, trainingSet) - gcn.dataset.labels[trainingSet, :])

    for l = gcn.numLayers:-1:1
        if l < gcn.numLayers
            dX = backpropagateActivationDerivative(gcn, gcn.activation,
                    gcn.reducedHiddenLayersBeforeActivation[l], dX)
        end
        Θ = gcn.weightMatrices[l,:]
        if l == 1
            for k = 1:gcn.numKernelParts
                gcn.weightMatrices[1,k] -= stepLength *
                    (gcn.reducedInputAfterKernels[k]' * dX
                        + gcn.architecture.regParam * Θ[k])
            end
        else
            for k = 1:gcn.numKernelParts
                gcn.weightMatrices[l,k] -= stepLength *
                    gcn.reducedHiddenLayers[l-1]' * (gcn.kernelDiagonals[k] .* dX)
            end
            dX = sum((gcn.kernelDiagonals[k] .* dX) * Θ[k]' for k in 1:gcn.numKernelParts)
        end
    end
    propagateLayers!(gcn)
end
