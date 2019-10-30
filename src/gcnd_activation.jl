
export
    applyActivation,
    backpropagateActivationDerivate,
    NoActivationMatrices,
    ReluMatrices,
    ReducedReluMatrices

"""
    applyActivation(gcn :: DirectGCN, act :: ActivationMatrices, X :: Matrix{Float64})

Apply the activation function to a matrix `X`. The implementation may depend on
the GCN type.
"""
applyActivation(:: DirectGCN, :: ActivationMatrices, X :: Matrix{Float64}) =
    error("No implementation available")

"""
    backpropagateActivationDerivative(gcn :: DirectGCN, act :: ActivationMatrices, X :: Matrix{Float64}, dY :: Matrix{Float64})

For an activation function `Y = σ(X)`, compute and return the gradient of `X`
with respect to some scalar objective function, based on `dY` the gradient of
`Y`.
"""
backpropagateActivationDerivative(:: DirectGCN, ::ActivationMatrices, X :: Matrix{Float64}, dY :: Matrix{Float64}) =
    error("No implementation available")



"""
    NoActivationMatrices

Subtype of `ActivationMatrices` for the `NoActivation` activation type.
"""
struct NoActivationMatrices <: ActivationMatrices end
ActivationMatrices(:: NoActivation) = NoActivationMatrices()

applyActivation(:: DirectGCN, ::NoActivationMatrices, X :: Matrix{Float64}) = X

backpropagateActivationDerivative(:: DirectGCN, ::NoActivationMatrices,
        :: Matrix{Float64}, dY :: Matrix{Float64}) = dY


"""
    ReluMatrices

Subtype of `ActivationMatrices` for the `Relu` activation type.
"""
struct ReluMatrices <: ActivationMatrices end
ActivationMatrices(:: Relu) = ReluMatrices()

function applyActivation(:: DirectStandardGCN, :: ReluMatrices, X :: Matrix{Float64})
    Y = copy(X)
    Y[X .< 0.0] .= 0.0
    return Y
end
function backpropagateActivationDerivative(:: DirectStandardGCN,
                :: ReluMatrices, X :: Matrix{Float64}, dY :: Matrix{Float64})
    dY[X .< 0.0] .= 0.0
    return dY
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


"""
    ReducedReluMatrices

Subtype of `ActivationMatrices` for the `ReducedRelu` activation type.
"""
struct ReducedReluMatrices <: ActivationMatrices end
ActivationMatrices(:: ReducedRelu) = ReducedReluMatrices()

function applyActivation(:: DirectLowRankGCN, ::ReducedReluMatrices, X :: Matrix{Float64})
    Y = copy(X)
    Y[X .< 0.0] .= 0.0
    return Y
end

function backpropagateActivationDerivative(:: DirectLowRankGCN,
            :: ReducedReluMatrices, X :: Matrix{Float64}, dX :: Matrix{Float64})
    dY[X .< 0.0] .= 0.0
    return dY
end
