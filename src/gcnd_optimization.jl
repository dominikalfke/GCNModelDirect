

export
    GradientDescentOptimizer,
    AdamOptimizer,
    optimizationStep!

##### Gradient Descent

mutable struct GradientDescentOptimizer
    learningRate :: Float64
    GradientDescentOptimizer(learningRate :: Float64 = 0.2) =
        new(learningRate)
end

function optimizationStep!(gcn :: DirectGCN, opt :: GradientDescentOptimizer)
    gradients = computeParameterGradients(gcn)
    updateParameters!(gcn, gradients, -opt.learningRate)
end


###### Momentum Gradient Descent

mutable struct MomentumGradientDescentOptimizer
    learningRate :: Float64
    momentum :: Float64

    MomentumGradientDescentOptimizer(learningRate :: Float64 = 0.1, momentum :: Float64 = 0.9) =
        momentum == 0.0 ? GradientDescentOptimizer(learningRate) : new(learningRate, momentum)
end

mutable struct MomentumGradientDescentOptimizerState
    velocities :: Matrix{Matrix{Float64}}
end

function optimizationStep!(gcn :: DirectGCN, opt :: MomentumGradientDescentOptimizer)
    gradients = computeParameterGradients(gcn)
    if isnothing(gcn.optimizerState)
        v = [(1-opt.momentum)*g for g in gradients]
        gcn.optimizerState = MomentumGradientDescentOptimizerState(v)
    else
        v = gcn.optimizerState.velocities
        for (index,g) in gradients
            axpby!(1-opt.momentum, g, opt.momentum, v[index])
        end
    end
    updateParameters!(gcn, v, -opt.learningRate)
end


##### Adam

mutable struct AdamOptimizer
    learningRate :: Float64
    β1 :: Float64
    β2 :: Float64
    ϵ :: Float64

    AdamOptimizer(learningRate :: Float64 = 0.001;
            β1 :: Float64 = 0.9, β2 :: Float64 = 0.999, ϵ :: Float64 = 1e-8) =
        new(learningRate, β1, β2, ϵ, 0, nothing, nothing)
end

mutable struct AdamOptimizerState
    timeStep :: Int
    firstMoments :: Matrix{Matrix{Float64}}
    secondMoments :: Matrix{Matrix{Float64}}
end


function optimizationStep!(gcn :: DirectGCN, opt :: AdamOptimizer)
    gradients = computeParameterGradients(gcn)
    L,K = size(gradients)
    if isnothing(gcn.optimizerState)
        st = AdamOptimizerState(1,
            [(1-opt.β1)*g for g in gradients],
            [(1-opt.β2)*g.^2 for g in gradients])
        gcn.optimizerState = st
    else
        st = gcn.optimizerState
        st.timeStep += 1
        for (index, g) in pairs(gradients)
            axpby!(1-opt.β1, g, opt.β1, st.firstMoments[index])
            axpby!(1-opt.β2, g.^2, opt.β2, st.secondMoments[index])
        end
    end
    t = st.timeStep
    updates = [st.firstMoments[l,k] ./ (sqrt.(st.secondMoments[l,k] .+ opt.ϵ))
                    for l=1:L, k=1:K]
    α = opt.learningRate * sqrt(1 - opt.β2^t) / (1 - opt.β1^t)
    updateParameters!(gcn, updates, -α)
end
