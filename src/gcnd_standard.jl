

abstract type StandardKernelMatrices <: KernelMatrices end

function DirectGCN(arc :: GCNArchitecture, dataset :: Dataset,
            kernel :: StandardKernelMatrices, activation :: ActivationMatrices)
    return DirectStandardGCN(arc, dataset, kernel, activation)
end

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

        return self
    end
end


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

output(gcn :: DirectStandardGCN, index :: Int) =
    applyKernelRows(gcn.kernel, gcn.outputBeforeKernels, index:index)[:]
output(gcn :: DirectStandardGCN, indexSet) =
    applyKernelRows(gcn.kernel, gcn.outputBeforeKernels, indexSet)
output(gcn :: DirectStandardGCN) =
    applyKernel(gcn.kernel, gcn.outputBeforeKernels)



function computeParameterGradients(gcn :: DirectStandardGCN)
    gradients = Matrix{Matrix{Float64}}(undef, gcn.numLayers, gcn.numKernelParts)

    trainingSet = gcn.dataset.trainingSet
    dX = (classProbabilities(gcn, trainingSet) - gcn.dataset.labels[trainingSet, :]) ./ length(trainingSet)

    KdX = applyKernelColumnsBeforeWeights(gcn.kernel, dX, trainingSet)
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



function gradientDescentStep!(gcn :: DirectStandardGCN; stepLength :: Float64 = 0.2)
    # trainingSet = gcn.dataset.trainingSet
    # dX = classProbabilities(gcn, trainingSet) - gcn.dataset.labels[trainingSet, :]
    # dX ./= length(trainingSet)
    #
    # KTdX = applyKernelColumnsBeforeWeights(gcn.kernel, dX, trainingSet)
    # Θ = gcn.weightMatrices[end,:]
    # for k in 1:gcn.numKernelParts
    #     gcn.weightMatrices[end,k] -= stepLength * (gcn.hiddenLayers[end]' * KTdX[k])
    # end
    #
    # for l = gcn.numLayers-1:-1:1
    #     dX = sum(KTdX[k] * Θ[k]' for k in 1:gcn.numKernelParts)
    #     dX = backpropagateActivationDerivative(gcn, gcn.activation,
    #                 gcn.hiddenLayersBeforeActivation[l], dX)
    #     Θ = gcn.weightMatrices[l,:]
    #     if l == 1
    #         for k in 1:gcn.numKernelParts
    #             gcn.weightMatrices[l,k] -= stepLength *
    #                 (gcn.inputAfterKernels[k]' * dX + gcn.architecture.regParam * Θ[k])
    #         end
    #     else
    #         KTdX = applyKernelBeforeWeights(gcn.kernel, dX)
    #         for k in 1:gcn.numKernelParts
    #             gcn.weightMatrices[l,k] -= stepLength * (gcn.hiddenLayers[l-1]' * KTdX[k])
    #         end
    #     end
    # end

    gradients = computeParameterGradients(gcn)
    for (index, g) in pairs(gradients)
        axpy!(-stepLength, g, gcn.weightMatrices[index])
    end

    propagateLayers!(gcn)
end
