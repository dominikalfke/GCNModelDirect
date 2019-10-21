
### FixedLowRankKernel

KernelMatrices(k :: FixedLowRankKernel) =
	LowRankKernelMatrices(k, size(k.projector, 2))

computeKernelMatrices(k :: FixedLowRankKernel, :: Dataset) =
	(k.projector, k.diagonals)

### LowRankPolyLaplacianKernel

KernelMatrices(k :: LowRankPolyLaplacianKernel) = LowRankKernelMatrices(k, k.rank)

computeKernelMatrices(k :: LowRankPolyLaplacianKernel, dataset :: Dataset) =
	setupMatrices(k, dataset)

### LowRankInvLaplacianKernel

KernelMatrices(k :: LowRankInvLaplacianKernel) = LowRankKernelMatrices(k, k.rank)

function computeKernelMatrices(k :: LowRankInvLaplacianKernel, dataset :: Dataset)
    U, d = setupMatrices(k, dataset)
	return U, [d]
end

### LowRankPolyHypergraphLaplacianKernel

KernelMatrices(k :: LowRankPolyHypergraphLaplacianKernel) = LowRankKernelMatrices(k, k.rank)

computeKernelMatrices(k :: LowRankPolyHypergraphLaplacianKernel, dataset :: Dataset) =
	setupMatrices(k, dataset)


### LowRankInvHypergraphLaplacianKernel

KernelMatrices(k :: LowRankInvHypergraphLaplacianKernel) = LowRankKernelMatrices(k, k.rank)

function computeKernelMatrices(k :: LowRankInvHypergraphLaplacianKernel, dataset :: Dataset)
    U, d = setupMatrices(k, dataset)
	return U, [d]
end
