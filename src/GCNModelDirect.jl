module GCNModelDirect

export
    DirectGCN,
    KernelMatrices,
    ActivationMatrices,
    computeKernelFactorization,
    classProbabilities,
    classPrediction,
    accuracy,
    runDirectExperiment

using GCNModel
using LinearAlgebra


abstract type DirectGCN end
abstract type KernelMatrices end
abstract type ActivationMatrices end

setupMatrices!(:: DirectGCN, :: KernelMatrices, :: Dataset) = nothing
setupMatrices!(:: DirectGCN, :: ActivationMatrices, :: Dataset) = nothing

include("gcnd_standard.jl")
include("gcnd_standard_kernels.jl")
include("gcnd_lowrank.jl")
include("gcnd_lowrank_kernels.jl")
include("gcnd_activation.jl")

DirectGCN(arc :: GCNArchitecture, dataset :: Dataset) =
    DirectGCN(arc, dataset,
        KernelMatrices(arc.kernel),
        ActivationMatrices(arc.activation))

function classProbabilities(gcn :: DirectGCN, index :: Int)
    y = exp.(output(gcn, index))
    return y ./ sum(y)
end
function classProbabilities(gcn :: DirectGCN, indexSet)
    Y = exp.(output(gcn, indexSet))
    return Y ./ sum(Y, dims=2)
end
function classProbabilities(gcn :: DirectGCN)
    Y = exp.(output(gcn))
    return Y ./ sum(Y, dims=2)
end

classPrediction(gcn :: DirectGCN, index :: Int) =
    argmax(output(gcn, index))

function accuracy(gcn :: DirectGCN, set = gcn.dataset.testSet)
    correctCount = 0.0
    for i in set
        correctCount += gcn.dataset.labels[i, classPrediction(gcn, i)]
    end
    return correctCount / length(set)
end


function runDirectExperiment(exp :: Experiment, numRuns :: Int; printInterval :: Int = 0)
    return repeatExperimentRuns(exp, numRuns, printInterval) do exp, dataset

        gcn, setupTime = @timed DirectGCN(exp.architecture, dataset)

        trainingTime = @elapsed for _ in 1:exp.numTrainingIter
            gradientDescentStep!(gcn)
        end

        acc = accuracy(gcn)

        return acc, setupTime, trainingTime
    end
end


end
