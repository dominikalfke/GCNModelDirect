
export
    setupActivation!,
    applyActivation,
    ReluMatrices,
    backpropagateActivationDerivate

setupActivation!(:: DirectGCN, :: ActivationMatrices, :: Dataset) = nothing

applyActivation(:: DirectGCN, :: ActivationMatrices, X :: Matrix{Float64}) =
    error("No implementation available")


struct ReluMatrices <: ActivationMatrices end
ActivationMatrices(:: Relu) = ReluMatrices()

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
