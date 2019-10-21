module GCNModelDirect

export
    DirectGCN,
    KernelMatrices,
    ActivationMatrices,
    computeKernelMatrices,
    runDirectExperiment

using GCNModel
using LinearAlgebra


abstract type DirectGCN end
abstract type KernelMatrices end
abstract type ActivationMatrices end

include("gcnd_lowrank.jl")
include("gcnd_lowrank_kernels.jl")
include("gcnd_activation.jl")

DirectGCN(arc :: GCNArchitecture, dataset :: Dataset) =
    DirectGCN(arc, dataset,
        KernelMatrices(arc.kernel),
        ActivationMatrices(arc.activation))


function runDirectExperiment(exp :: Experiment, numRuns :: Int; printInterval :: Int = 0)
    repeatExperimentRuns(exp, numRuns, printInterval) do exp, dataset

        gcn, setupTime = @timed DirectGCN(exp.architecture, dataset)

        trainingTime = @elapsed for _ in 1:exp.numTrainingIter
            gradientDescentStep!(gcn)
        end

        acc = accuracy(gcn)

        return acc, setupTime, trainingTime
    end

end


end
