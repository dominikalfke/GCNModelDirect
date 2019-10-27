
### FixedLowRankKernel

KernelMatrices(k :: FixedLowRankKernel) =
	LowRankKernelMatrices(k, size(k.projector, 2))

computeKernelFactorization(k :: FixedLowRankKernel, :: Dataset) =
	(k.projector, k.diagonals)

### LowRankPolyLaplacianKernel

KernelMatrices(k :: LowRankPolyLaplacianKernel) = LowRankKernelMatrices(k, k.rank)

computeKernelFactorization(k :: LowRankPolyLaplacianKernel, dataset :: Dataset) =
	computeMatrices(k, dataset)

### LowRankInvLaplacianKernel

KernelMatrices(k :: LowRankInvLaplacianKernel) = LowRankKernelMatrices(k, k.rank)

function computeKernelFactorization(k :: LowRankInvLaplacianKernel, dataset :: Dataset)
    U, d = computeMatrices(k, dataset)
	return U, [d]
end

### LowRankPolyHypergraphLaplacianKernel

KernelMatrices(k :: LowRankPolyHypergraphLaplacianKernel) = LowRankKernelMatrices(k, k.rank)

computeKernelFactorization(k :: LowRankPolyHypergraphLaplacianKernel, dataset :: Dataset) =
	computeMatrices(k, dataset)


### LowRankInvHypergraphLaplacianKernel

KernelMatrices(k :: LowRankInvHypergraphLaplacianKernel) = LowRankKernelMatrices(k, k.rank)

function computeKernelFactorization(k :: LowRankInvHypergraphLaplacianKernel, dataset :: Dataset)
    U, d = computeMatrices(k, dataset)
	return U, [d]
end
