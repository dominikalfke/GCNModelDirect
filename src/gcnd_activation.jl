
export
    setupMatrices!,
    applyActivation,
    ReluMatrices,
    backpropagateActivationDerivate

applyActivation(:: DirectGCN, :: ActivationMatrices, X :: Matrix{Float64}) =
    error("No implementation available")


struct ReluMatrices <: ActivationMatrices end
ActivationMatrices(:: Relu) = ReluMatrices()

function applyActivation(gcn :: DirectStandardGCN, :: ReluMatrices, X :: Matrix{Float64})
    X = copy(X)
    X[X .< 0.0] .= 0.0
    return X
end
function backpropagateActivationDerivative(gcn :: DirectStandardGCN,
                :: ReluMatrices, X :: Matrix{Float64}, dX :: Matrix{Float64})
    dX[X .< 0.0] .= 0.0
    return dX
end

function applyActivation(gcn :: DirectLowRankGCN, :: ReluMatrices, X :: Matrix{Float64})
    Y = gcn.reductionMatrix * X
    Y[Y .< 0.0] .= 0.0
    return gcn.reductionMatrix' * Y
end

function backpropagateActivationDerivative(gcn :: DirectLowRankGCN,
            :: ReluMatrices, X :: Matrix{Float64}, dX :: Matrix{Float64})
    maskedUdX = gcn.reductionMatrix * dX
    maskedUdX[(gcn.reductionMatrix * X) .< 0.0] .= 0.0
    return gcn.reductionMatrix' * maskedUdX
end
