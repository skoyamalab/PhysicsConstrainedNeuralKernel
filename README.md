# PhysicsConstrainedNeuralKernel
Sound field estimation based on physics-constrained neural kernels.

This is a library for performing kernel interpolation using adaptive kernel functions that are always guaranteed to satisfy the Helmholtz equation, meaning they can be used to interpolate sound field pressure values without requiring any form of enforcement and can be trained using data-driven methods alone. The entire library is written in Julia and the kernel functions shown here were previously proposed in [these](https://doi.org/10.48550/arXiv.2408.14731) [works](10.36227/techrxiv.24455380.v2).

## Compiling the library

In order to use the library, it is necessary to add the following header (CUDA is optional and only relevant if there is a NVIDIA gpu):
```
using GenericLinearAlgebra, LinearAlgebra, Polynomials, SpecialFunctions, ArrayAllocators, SphericalHarmonics, KernelFunctions, ChainRules,Lebedev, Functors, Optimization, Optimisers, FFTW, Flux, KernelAbstractions, OrdinaryDiffEq
using CUDA
using Tullio
using OrdinaryDiffEq, SciMLSensitivity
DIR = @__DIR__
include(string(DIR, "/PCNK/master.jl"))
using .PCNK
```
Once the necessary packages have been loaded (given that the PCNK folder is in the same directory), the kernels can be instantiated by offering the constructors the necessary data, thoroughly commented in the source files.

## Using the kernels

All kernels support 4 patterns of input: point-to-point, point-to-batch, batch-to-point and batch-to-batch. Batched inputs are matrices where each column is a different vector.

Vector-to-vector is the standard input for a kernel function: it is simply the evaluation of the kernel on the specified positions and results on a scalar:

$$
\kappa (\mathbf{x}, \mathbf{x}^\prime) \in \mathbb{C}
$$

Vector-to-batch is the evaluation of a vector against each column of the batch, and results on a vector:


$$
\kappa (\mathbf{x}, \mathbf{X} ) = \begin{bmatrix} \kappa(\mathbf{x}, \mathbf{x}_1) \\\ \vdots \\\ \kappa(\mathbf{x}, \mathbf{x}_N)\end{bmatrix},
$$

and likewise batch-to-vector is a vector where each column of the batch operates with the vector

$$
\kappa (\mathbf{X}, \mathbf{x} ) = \begin{bmatrix} \kappa(\mathbf{x}_1, \mathbf{x}) \\\ \vdots \\\ \kappa(\mathbf{x}_N, \mathbf{x})\end{bmatrix}.
$$

Finally, batch-to-batch results in a full matrix where rows are iterated on the first batch, while columns are iterated on the second:

$$
\kappa(\mathbf{X}, \mathbf{X}^\prime) = \begin{bmatrix} \kappa(\mathbf{x}_1, \mathbf{x}_1^\prime) & \kappa(\mathbf{x}_1, \mathbf{x}_2^\prime) & \dots & \kappa(\mathbf{x}_1, \mathbf{x}_M^\prime) \\\ \kappa(\mathbf{x}_2, \mathbf{x}_1^\prime) & \kappa(\mathbf{x}_2, \mathbf{x}_2^\prime) & \dots & \kappa(\mathbf{x}_2, \mathbf{x}_M^\prime) \\\ \vdots & \vdots & \ddots & \vdots \\\ \kappa(\mathbf{x}_N, \mathbf{x}_1^\prime) & \kappa(\mathbf{x}_N, \mathbf{x}_2^\prime) & \dots & \kappa(\mathbf{x}_N, \mathbf{x}_M^\prime) \end{bmatrix}.
$$

And of course, the Gram matrix (very important for kernel methods) is the application of a batch with itself.

$$
\mathrm{Gram}(\kappa, \mathbf{X}) = \kappa(\mathbf{X}, \mathbf{X}) = \begin{bmatrix} \kappa(\mathbf{x}_1, \mathbf{x}_1) & \kappa(\mathbf{x}_1, \mathbf{x}_2) & \dots & \kappa(\mathbf{x}_1, \mathbf{x}_N) \\\ \kappa(\mathbf{x}_2, \mathbf{x}_1) & \kappa(\mathbf{x}_2, \mathbf{x}_2) & \dots & \kappa(\mathbf{x}_2, \mathbf{x}_N) \\\ \vdots & \vdots & \ddots & \vdots \\\ \kappa(\mathbf{x}_N, \mathbf{x}_1) & \kappa(\mathbf{x}_N, \mathbf{x}_2) & \dots & \kappa(\mathbf{x}_N, \mathbf{x}_N) \end{bmatrix}.
$$

All of these operations are implemented without scalar indexing or mutations and have been tested on Nvidia gpus. 

## Types of kernels available

The library supports several kernel models, [explained](10.1109/LSP.2017.2775242) [in depth](10.1109/ICASSP40776.2020.9053416) [in these](10.1109/WASPAA52581.2021.9632731) [papers](10.36227/techrxiv.24455380.v2). The instantiation of each kernel is explained in the source files in PCNK/src/Kernels.

## Hands-on example

The file SPM.jl shows how to instantiate and train our kernels by performing interpolation on a simulated sound created to have reverberation time of $400~\mathrm{ms}$. We contrast a fully adaptive physics-constrained neural kernel proposed in [this paper](https://doi.org/10.48550/arXiv.2408.14731) with a uniform kernel that does no training whatsoever. The results of the comparison run can be seen in the 400ms/GRAPHS folder.
