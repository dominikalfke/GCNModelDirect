
export
    DirectGCN,
    KernelMatrices,
    ActivationMatrices,
    setupMatrices!,
    classProbabilities,
    classPrediction,
    accuracy,
    runDirectExperiment

"""
    DirectGCN

Abstract supertype for implementations of GCN.
"""
abstract type DirectGCN end


"""
    KernelMatrices

Abstract supertype whose subtypes hold kernel-specific matrices.
"""
abstract type KernelMatrices end

"""
    KernelMatrices(k :: GCNKernel)

Create an instance of a KernelMatrices subtype tailored for the given kernel type.
"""
KernelMatrices(k :: GCNKernel) =
    error("No KernelMatrices implementation available for kernels of type $(typeof(k))")


"""
    ActivationMatrices

Abstract supertype whose subtypes hold activation-specific matrices.
"""
abstract type ActivationMatrices end

"""
    ActivationMatrices(k :: GCNKernel)

Create an instance of a ActivationMatrices subtype tailored for the given kernel type.
"""
ActivationMatrices(act :: Activation) =
    error("No ActivationMatrices implementation available for activation of type $(typeof(act))")


"""
    setupMatrices!(:: DirectGCN, :: KernelMatrices, :: Dataset)

Fill the fields of the given KernelMatrices subtype with data based on the
dataset and the created DirectGCN subtype.
"""
setupMatrices!(:: DirectGCN, :: KernelMatrices, :: Dataset) = nothing
setupMatrices!(:: DirectGCN, :: ActivationMatrices, :: Dataset) = nothing

"""
    DirectGCN(arc :: GCNArchitecture, dataset :: Dataset)
    DirectGCN(arc :: GCNArchitecture, dataset :: Dataset, kernel :: KernelMatrices, act :: ActivationMatrices)

Construct an instance of a DirectGCN subtype that gives an implementation of the
given architecture for the given dataset. The returned DirectGCN subtype depends
on the kernel and activation.
"""
DirectGCN(arc :: GCNArchitecture, dataset :: Dataset) =
    DirectGCN(arc, dataset,
        KernelMatrices(arc.kernel),
        ActivationMatrices(arc.activation))


"""
    output(gcn :: DirectGCN)
    output(gcn :: DirectGCN, index :: Int)
    output(gcn :: DirectGCN, indexSet)

Produce the output matrix of the last GCN layer. If an index set is given, only
the submatrix of indicated rows is returned. If a single index is given, only
the indicated row is returned as a column vector.
"""
output(:: DirectGCN, index=0) =
    error("No output defined for this DirectGCN implementation")

"""
    classProbabilities(gcn :: DirectGCN)
    classProbabilities(gcn :: DirectGCN, index :: Int)
    classProbabilities(gcn :: DirectGCN, indexSet)

Produce the matrix of class probabilities based on the GCN, i.e., the softmax
function applied to the rows of the output matrix. If an index set is given,
only the submatrix of indicated rows is returned. If a single index is given,
only the vector of probabilities for that one sample is returned.
"""
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

"""
    classPrediction(gcn :: DirectGCN, index :: Int)

Return the index of the class which is predicted to be most likely for the
sample with the given index.
"""
classPrediction(gcn :: DirectGCN, index :: Int) =
    argmax(output(gcn, index))

"""
    accuracy(gcn :: DirectGCN)
    accuracy(gcn :: DirectGCN, indexSet)

Return the ratio of samples in the given set whose class is predicted correctly,
based on the true labels in the dataset object. If no set is given, the dataset
test set is used (as opposed to the full data set).
"""
function accuracy(gcn :: DirectGCN, set = gcn.dataset.testSet)
    correctCount = 0.0
    for i in set
        correctCount += gcn.dataset.labels[i, classPrediction(gcn, i)]
    end
    return correctCount / length(set)
end

"""
    runDirectExperiment(exp :: Experiment, numRuns :: Int; printInterval = 0)
    runDirectExperiment(exp :: Experiment, numRuns :: Int, optimizer; printInterval = 0)

Perform runs of the given experiment setting using a direct implementation, and
store the results in the Experiment object. The optimizer can be any object for
which `optimizationStep!(gcn, optimizer)` is defined. If no optimizer is given,
a `GradientDescentOptimizer` with default learning rate is used.
If a nonzero `printInterval` is given, an average accuracy is printed every that
many runs.
"""
function runDirectExperiment(exp :: Experiment, numRuns :: Int,
                                opt = GradientDescentOptimizer();
                                printInterval :: Int = 0)
    return repeatExperimentRuns(exp, numRuns, printInterval) do exp, dataset

        gcn, setupTime = @timed DirectGCN(exp.architecture, dataset)

        trainingTime = @elapsed for _ in 1:exp.numTrainingIter
            optimizationStep!(gcn, opt)
        end

        acc = accuracy(gcn)

        return acc, setupTime, trainingTime
    end
end


"""
    computeParameterGradients(gcn :: DirectGCN)

Compute the gradients of all parameters of a GCN with respect to the loss on the
training set.
"""
computeParameterGradients(gcn :: DirectGCN) =
    error("No computeParameterGradients implementation available for GCN of type $(typeof(gcn))")

"""
    updateParameters!(gcn :: DirectGCN, dΘ)
    updateParameters!(gcn :: DirectGCN, dΘ, factor)

Update the parameters in a GCN by adding (possibly scaled) values to them. The
structure of the update `dΘ` depends on the GCN type, but should generally be
the same as the return value of `computeParameterGradients`.
"""
updateParameters!(gcn :: DirectGCN, dΘ, factor :: Float64 = 1.0)
