module GCNModelDirect

using GCNModel
using LinearAlgebra

include("gcnd_general.jl")
include("gcnd_standard.jl")
include("gcnd_standard_kernels.jl")
include("gcnd_lowrank.jl")
include("gcnd_lowrank_kernels.jl")
include("gcnd_activation.jl")
include("gcnd_optimization.jl")

end
