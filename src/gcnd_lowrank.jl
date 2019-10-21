

export
    LowRankKernelMatrices,
    DirectLowRankGCN,
    initializeRandomWeights!,
    propagateLayers!,
    output,
    classProbabilities,
    classPrediction,
    accuracy,
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

    reducedInput :: Matrix{Float64}
    reducedHiddenLayersBeforeActivation :: Vector{Matrix{Float64}}
    reducedHiddenLayers :: Vector{Matrix{Float64}}
    reducedOutput :: Matrix{Float64}
    trueClasses :: Vector{Int}

    function DirectLowRankGCN(arc :: GCNArchitecture, dataset :: Dataset,
                rank :: Int, activation :: ActivationMatrices)

        self = new(arc, dataset, arc.kernel, rank, activation)

        checkCompatibility(arc, dataset)

        self.numLayers = length(arc.layerWidths)-1
        self.numKernelParts = numParts(arc.kernel)

        self.reductionMatrix, self.kernelDiagonals = computeKernelMatrices(self.kernel, dataset)
        setupActivation!(self, self.activation, dataset)

        self.weightMatrices = Matrix{Matrix{Float64}}(undef, self.numLayers, self.numKernelParts)
        initializeRandomWeights!(self)

        self.reducedInput = self.reductionMatrix' * dataset.features
        self.reducedHiddenLayersBeforeActivation = Vector{Matrix{Float64}}(undef, self.numLayers-1)
        self.reducedHiddenLayers = Vector{Matrix{Float64}}(undef, self.numLayers-1)
        propagateLayers!(self)

        self.trueClasses = [argmax(dataset.labels[i,:]) for i in 1:dataset.numNodes]

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
    X = gcn.reducedInput
    for l = 1:gcn.numLayers
        X = sum(gcn.kernelDiagonals[k] .* (X * gcn.weightMatrices[l,k])
                for k = 1:gcn.numKernelParts)
        if l == gcn.numLayers
            gcn.reducedOutput = X
        else
            gcn.reducedHiddenLayersBeforeActivation[l] = X
            X = applyActivation(gcn, gcn.activation, X)
            gcn.reducedHiddenLayers[l] = X
        end
    end
end

output(gcn :: DirectLowRankGCN, index :: Int) =
    gcn.reducedOutput' * gcn.reductionMatrix[index, :]
output(gcn :: DirectLowRankGCN, set = 1:gcn.dataset.numNodes) =
    gcn.reductionMatrix[set, :] * gcn.reducedOutput

function classProbabilities(gcn :: DirectLowRankGCN, index :: Int)
    y = exp.(output(gcn, index))
    return y ./ sum(y)
end
function classProbabilities(gcn :: DirectLowRankGCN, set = 1:gcn.dataset.numNodes)
    Y = exp.(output(gcn, set))
    return Y ./ sum(Y, dims=2)
end

classPrediction(gcn :: DirectLowRankGCN, index :: Int) =
    argmax(output(gcn, index))

function accuracy(gcn :: DirectLowRankGCN, set = gcn.dataset.testSet)
    correctCount = 0
    for i in set
        correctCount += (gcn.trueClasses[i] == classPrediction(gcn, i))
    end
    return correctCount / length(set)
end



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
                    (gcn.reducedInput' * (gcn.kernelDiagonals[k] .* dX)
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
