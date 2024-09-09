# ====== Source file for the composite kernel implementations ========
# === These kernels combine two different kernel implementations ====

#These are the kernels that are composed of two kernel types together.

abstract type CompositeKernel{T<:AbstractFloat} <: ISFKernel{T} end

# Our kernel implementation, which mixes an infinite-dimensional analytical Bessel kernel with a neural weight kernel using plane waves.

mutable struct DirectedResidualKernel{T<:AbstractFloat, N<:Integer} <: CompositeKernel{T}
    AnalyticalKernel::MultiDirectionalKernel{T}
    NeuralKernel::NeuralWeightPlaneWaveKernel{T, N}
end

# Constructors of the kernel require not only a wave number, but also orders for the analytical and neural components.

function DirectedResidualKernel{T, N}(k::U, Ord1::Integer, Ord2::Integer) where {T<:AbstractFloat, N<:Integer, U<:Number}
    return DirectedResidualKernel{T, N}(MultiDirectionalKernel{T}(k, Ord1), NeuralWeightPlaneWaveKernel{T, N}(k, Ord2))
end

function DirectedResidualKernel{T}(k::U, Ord1::N, Ord2::Integer) where {T<:AbstractFloat, N<:Integer, U<:Number}
    return DirectedResidualKernel{T, N}(MultiDirectionalKernel{T}(k, Ord1), NeuralWeightPlaneWaveKernel{T, N}(k, Ord2))
end

function DirectedResidualKernel(k::T, Ord1::N, Ord2::Integer) where {T<:AbstractFloat, N<:Integer}
    return DirectedResidualKernel{T, N}(MultiDirectionalKernel{T}(k, Ord1), NeuralWeightPlaneWaveKernel{T, N}(k, Ord2))
end

# Implementation of a fully plane-wave-based kernel mixing an analytical plane wave kernel with a fixed number of directions with an auxilliary neural kernel.

mutable struct PlaneWaveCompositeKernel{T<:AbstractFloat, N<:Integer} <: CompositeKernel{T}
    AnalyticalKernel::PlaneWaveKernel{T}
    NeuralKernel::NeuralWeightPlaneWaveKernel{T, N}
end

# Constructors require the same as the previous case.

function PlaneWaveCompositeKernel{T, N}(k::U, Ord1::Integer, Ord2::Integer) where {T<:AbstractFloat, N<:Integer, U<:Number}
    return PlaneWaveCompositeKernel{T, N}(PlaneWaveKernel{T}(k, Ord1), NeuralWeightPlaneWaveKernel{T, N}(k, Ord2))
end

function PlaneWaveCompositeKernel{T}(k::U, Ord1::N, Ord2::Integer) where {T<:AbstractFloat, N<:Integer, U<:Number}
    return PlaneWaveCompositeKernel{T, N}(PlaneWaveKernel{T}(k, Ord1), NeuralWeightPlaneWaveKernel{T, N}(k, Ord2))
end

function PlaneWaveCompositeKernel(k::T, Ord1::N, Ord2::Integer) where {T<:AbstractFloat, N<:Integer}
    return PlaneWaveCompositeKernel{T, N}(PlaneWaveKernel{T}(k, Ord1), NeuralWeightPlaneWaveKernel{T, N}(k, Ord2))
end

# Implementations (thankfully it works for both)

function (a::CompositeKernel)(ΔX::AbstractArray{<:Number})
    return a.AnalyticalKernel(ΔX) + a.NeuralKernel(ΔX)
end

function (a::CompositeKernel)(x1::AbstractVecOrMat{<:Number}, x2::AbstractVecOrMat{<:Number})
    return a.AnalyticalKernel(x1, x2) + a.NeuralKernel(x1, x2)
end

# Flux management
@functor DirectedResidualKernel
@functor PlaneWaveCompositeKernel
trainable(a::CompositeKernel) = (; a.AnalyticalKernel, a.NeuralKernel)