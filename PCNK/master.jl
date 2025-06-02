module PCNK
dir = @__DIR__
src = string(dir, "/src")
using GenericLinearAlgebra, LinearAlgebra, SpecialFunctions, ArrayAllocators, KernelFunctions, ChainRules, JLD2,Lebedev, Functors, Optimisers, FFTW, Flux, KernelAbstractions, OrdinaryDiffEq, SciMLSensitivity, Distributions
#Uncomment if not using CUDA. While the code is agnostic to it, CUDA is the only GPU framework the library has been tested on.
using KernelFunctions: Kernel
using ChainRules: @ignore_derivatives, @scalar_rule
using Functors: @functor
using Optimisers: Restructure
import Optimisers: trainable
using ForwardDiff
using ForwardDiff: Dual, value, partials
import Flux: Chain

export ISFKernel, UniformKernel, PlaneWaveKernel, DirectionalKernel,
       MultiDirectionalKernel, NeuralWeightPlaneWaveKernel, DirectedResidualKernel, PlaneWaveCompositeKernel, NeuralAugmentedKernel,
       resample, j0

abstract type ISFKernel{T<:AbstractFloat} <: KernelFunctions.Kernel end # Interior sound field kernels

abstract type NeuralISFKernel{T<:AbstractFloat, N<:Integer} <: ISFKernel{T} end # ISFKernel embedded with a neural network


include(joinpath(src, "Background", "_background.jl"))


# Interior sound field kernel implementations

ISFK_src = joinpath(src, "Kernels")

include(joinpath(ISFK_src, "UniformISFK","_uniformisfk.jl"))
include(joinpath(ISFK_src, "DirectionalISFK", "_directionalisfk.jl"))
include(joinpath(ISFK_src, "PlaneWaveISFK","_planewaveisfk.jl"))
include(joinpath(ISFK_src, "PhysicsInformedNeuralISFK", "_physicsinformedneuralisfk.jl"))

# Miscellaneous functions and related resources for the various kernels

Misc_src = joinpath(src, "Miscellaneous")

include(joinpath(Misc_src, "KernelResampler.jl"))
include(joinpath(Misc_src, "NNODE.jl"))

end