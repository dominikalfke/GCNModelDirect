module GCNModelDirect

export
    DirectGCN,
    KernelMatrices,
    ActivationMatrices,
    computeKernelMatrices

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

end
