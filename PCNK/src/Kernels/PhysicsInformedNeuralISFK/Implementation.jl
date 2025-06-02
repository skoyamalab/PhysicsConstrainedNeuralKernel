abstract type CompositeKernel{T<:AbstractFloat} <: ISFKernel{T} end

# Standard physics-informed plane wave kernel with the weight given by a neural network 
mutable struct PlaneWavePINKernel{T<:AbstractFloat} <: CompositeKernel{T}
    AnalyticalKernel::scaled_UniformKernel{T}
    NeuralKernel::NeuralWeightPlaneWaveKernel{T}
end

# Non-generic constructors:

function PlaneWavePINKernel{T}(k::U, Ord::Integer, W=__PWISFK_W0) where {T<:AbstractFloat, U<:Number}
    AnalyticalKernel = scaled_UniformKernel{T}(k)
    NeuralKernel = NeuralWeightPlaneWaveKernel{T}(k, Ord, W = W)
    return PlaneWavePINKernel{T}(AnalyticalKernel, NeuralKernel)
end

function PlaneWavePINKernel(k::U, Ord::Integer, W=__PWISFK_W0) where {U<:Number}
    T = float(U)
    AnalyticalKernel = scaled_UniformKernel{T}(k)
    NeuralKernel = NeuralWeightPlaneWaveKernel{T}(k, Ord, W = W)
    return PlaneWavePINKernel{T}(AnalyticalKernel, NeuralKernel)
end

# Complete direct and residual model physics-informed kernel function.
mutable struct DirectedResidualPINKernel{T<:AbstractFloat} <: CompositeKernel{T}
    AnalyticalKernel::DirectionalSFKernel{T, <:AbstractVecOrMat}
    NeuralKernel::NeuralWeightPlaneWaveKernel{T}
end

function DirectedResidualPINKernel{T}(k::U, Ord_dir::Integer, Ord_res::Integer, W = __PWFISFK_W0; trainable_direction::Bool = false) where {T<:AbstractFloat, U<:Real}
    AnalyticalKernel = DirectionalSFKernel{T}(k, Ord_dir, trainable_direction = trainable_direction)
    NeuralKernel = NeuralWeightPlaneWaveKernel{T}(k, Ord_res, W=W)
    return DirectedResidualPINKernel{T}(AnalyticalKernel, NeuralKernel)
end

function DirectedResidualPINKernel(k::U, Ord_dir::Integer, Ord_res::Integer, W = __PWFISFK_W0; trainable_direction::Bool = false) where {U<:Real}
    T = float(U)
    AnalyticalKernel = DirectionalSFKernel{T}(k, Ord_dir, trainable_direction = trainable_direction)
    NeuralKernel = NeuralWeightPlaneWaveKernel{T}(k, Ord_res, W=W)
    return DirectedResidualPINKernel{T}(AnalyticalKernel, NeuralKernel)
end

@functor PlaneWavePINKernel
@functor DirectedResidualPINKernel
Flux.@layer PlaneWavePINKernel
Flux.@layer DirectedResidualPINKernel