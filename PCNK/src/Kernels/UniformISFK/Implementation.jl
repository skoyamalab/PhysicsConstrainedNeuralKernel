# ====== Source file for the uniform weight kernel implementation ========
# ===== Weight equivalent to considering directionally uniform gain ======

abstract type UniformKernel{T<:AbstractFloat} <: ISFKernel{T} end

# Struct definition, first predicting a lack of trainable multiplicative factor

mutable struct fixed_UniformKernel{T} <:UniformKernel{T}
    k::T
    function fixed_UniformKernel{T}(k::U) where {U<:Real, T<:AbstractFloat}
        return new(T(k))
    end
    function fixed_UniformKernel(k::U) where {U<:Real}
        T = float(U)
        return fixed_UniformKernel{T}(k)
    end
end

@functor fixed_UniformKernel
trainable(::fixed_UniformKernel) = (;)
Flux.@layer fixed_UniformKernel

# Then we also include a case with a fixed multiplicative factor for the combination of different kernels.

mutable struct scaled_UniformKernel{T} <:UniformKernel{T}
    k::T
    σ::AbstractVector
end

function scaled_UniformKernel{T}(k::U; σ::Real = 1.) where {U<:Real, T<:AbstractFloat}
    return scaled_UniformKernel{T}(T(k), T[σ])
end
function scaled_UniformKernel(k::U; σ::Real = 1.) where {U<:Real}
    T = float(U)
    return scaled_UniformKernel{T}(k, σ = σ)
end

@functor scaled_UniformKernel
trainable(a::scaled_UniformKernel) = (; σ = a.σ)
Flux.@layer scaled_UniformKernel

# Generic constructor that accounts for both cases:

function UniformKernel{T}(k::Real; σ::Union{Nothing, Real} = nothing) where {T<:AbstractFloat}
    if isnothing(σ)
        return fixed_UniformKernel{T}(k)
    else
        return scaled_UniformKernel{T}(k, σ = σ)
    end
end

function UniformKernel(k::Real; σ::Union{Nothing, Real} = nothing)
    if isnothing(σ)
        return fixed_UniformKernel(k)
    else
        return scaled_UniformKernel(k, σ = σ)
    end
end

