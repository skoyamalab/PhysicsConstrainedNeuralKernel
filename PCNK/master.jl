module PCNN
dir = @__DIR__
src = string(dir, "/src")
using GenericLinearAlgebra, LinearAlgebra, Polynomials, SpecialFunctions, ArrayAllocators, SphericalHarmonics, KernelFunctions, ChainRules, JLD2,Lebedev, Functors, Optimisers, FFTW, Flux, KernelAbstractions
#Uncomment if not using CUDA. While the code is agnostic to it, CUDA is the only GPU framework the library has been tested on.
using CUDA
using Tullio
using KernelFunctions: Kernel
using ChainRules: @ignore_derivatives, @scalar_rule
using Functors: @functor
import Optimisers: trainable

export ISFKernel, UniformKernel, PlaneWaveKernel, DirectionalKernel,
       MultiDirectionalKernel, NeuralWeightPlaneWaveKernel, DirectedResidualKernel, PlaneWaveCompositeKernel,
       resample, j0

abstract type ISFKernel{T<:AbstractFloat} <: KernelFunctions.Kernel end # Interior sound field kernels

abstract type NeuralISFKernel{T<:AbstractFloat, N<:Integer} <: ISFKernel{T} end # ISFKernel embedded with a neural network


BG_src = string(src, "/Background")
include(string(BG_src, "/SphericalBesselImpl.jl"))

# Interior sound field kernel implementations

ISFK_src = string(src, "/Kernels")

include(string(ISFK_src, "/UniformISFK.jl"))
include(string(ISFK_src, "/DirectionalISFK.jl"))
include(string(ISFK_src, "/PlaneWaveISFK.jl"))
include(string(ISFK_src, "/MultiDirectionalISFK.jl"))
include(string(ISFK_src, "/CompositeISFK.jl"))

# Miscellaneous functions and related resources for the various kernels

include(string(BG_src, "/KernelResampler.jl"))

end