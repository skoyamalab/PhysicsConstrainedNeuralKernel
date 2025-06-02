# Generic object for the directionally biased spherical Bessel J kernels
abstract type DirectionalSFKernel{T<:AbstractFloat, V<:AbstractVecOrMat} <: ISFKernel{T} end

#Kernel function with fixed directions
mutable struct FixedDirectionSFKernel{T<:AbstractFloat, V<:AbstractVecOrMat} <: DirectionalSFKernel{T, V}
    k::T
    σ::AbstractVector
    β::AbstractVector
    v::V
end

# Non-generic constructors
# You can inform an integration order for a Lebedev grid
function FixedDirectionSFKernel{T}(k::Real, Ord::Integer; σ::Union{Nothing, Real} = nothing) where {T<:AbstractFloat}
    x,y,z, w = lebedev_by_order(Ord)
    v =[x y z]'
    v = Matrix{T}(v ./ sum(abs2, v, dims=1))
    if isnothing(σ)
        _σ = T.(w)
    else
        _σ = T.(abs(σ) * w)
    end
    β = fill(T(10), length(w))
    return FixedDirectionSFKernel{T, typeof(v)}(T(k), _σ, β, v)
end
# Or a set of directions outright
function FixedDirectionSFKernel{T}(k::Real, v::AbstractMatrix{<:Real}; σ::Union{Nothing, Real, AbstractVector{<:Real}} = nothing) where {T<:AbstractFloat}
    N = size(v, 2)
    σ_dist = Dirichlet(ones(N))
    if isnothing(σ)
        _σ = T.(rand(σ_dist))
    elseif isa(σ, Real)
        _σ = T.(abs(σ) * rand(σ_dist))
    else
        if length(σ) != N
            error("The number of weights has to correspond to the number of directions!")
        else
            _σ = T.(abs.(σ))
        end
    end
    _v = T.(v)
    β = fill(T(10), size(v,2))
    return FixedDirectionSFKernel{T, typeof(_v)}(T(k), _σ, β, _v)
end
# Or a single direction
function FixedDirectionSFKernel{T}(k::Real, v::AbstractVector{<:Real}; σ::Union{Nothing, Real} = nothing) where {T<:AbstractFloat}
    if isnothing(σ)
        _σ = T[one]
    else
        _σ = T[abs(σ)]
    end
    _v = T.(v)
    β = T[10]
    return FixedDirectionSFKernel{T, typeof(_v)}(T(k), _σ, β, v)
end
# There are also constructors where the type is infered:
function FixedDirectionSFKernel(k::Real, Ord::Integer; σ::Union{Nothing, Real} = nothing)
    T = eltype(k)
    return FixedDirectionSFKernel{T}(k, Ord, σ = σ)
end
function FixedDirectionSFKernel(k::Real, v::AbstractMatrix{<:Real}; σ::Union{Nothing, Real, AbstractVector{<:Real}} = nothing)
    T = float(eltype(k))
    return FixedDirectionSFKernel{T}(k, v, σ = σ)
end
function FixedDirectionSFKernel(k::Real, v::AbstractVector{<:Real}; σ::Union{Nothing, Real} = nothing)
    T = float(eltype(k))
    return FixedDirectionSFKernel{T}(k, v, σ = σ)
end

# Flux management
@functor FixedDirectionSFKernel
Flux.@layer FixedDirectionSFKernel
trainable(a::FixedDirectionSFKernel) = (; σ = a.σ, a.β)

#Kernel with trainable direction(s)
mutable struct TrainableDirectionSFKernel{T<:AbstractFloat, V<:AbstractVecOrMat} <: DirectionalSFKernel{T, V}
    k::T
    σ::AbstractVector
    β::AbstractVector
    v::V
end


function TrainableDirectionSFKernel{T}(k::Real, Ord::Integer; σ::Union{Nothing, Real} = nothing) where {T<:AbstractFloat}
    x,y,z, w = lebedev_by_order(Ord)
    v =[x y z]'
    v = Matrix{T}(v ./ sum(abs2, v, dims=1))
    if isnothing(σ)
        _σ = T.(w)
    else
        _σ = T.(abs(σ) * w)
    end
    β = fill(T(10), length(w))
    return TrainableDirectionSFKernel{T, typeof(v)}(T(k), _σ, β, v)
end
# Or a set of directions outright
function TrainableDirectionSFKernel{T}(k::Real, v::AbstractMatrix{<:Real}; σ::Union{Nothing, Real, AbstractVector{<:Real}} = nothing) where {T<:AbstractFloat}
    N = size(v, 2)
    σ_dist = Dirichlet(ones(N))
    if isnothing(σ)
        _σ = T.(rand(σ_dist))
    elseif isa(σ, Real)
        _σ = T.(abs(σ) * rand(σ_dist))
    else
        if length(σ) != N
            error("The number of weights has to correspond to the number of directions!")
        else
            _σ = T.(abs.(σ))
        end
    end
    _v = T.(v)
    β = fill(T(10), size(v,2))
    return TrainableDirectionSFKernel{T, typeof(_v)}(T(k), _σ, β, _v)
end
# Or a single direction
function TrainableDirectionSFKernel{T}(k::Real, v::AbstractVector{<:Real}; σ::Union{Nothing, Real} = nothing) where {T<:AbstractFloat}
    if isnothing(σ)
        _σ = T[one]
    else
        _σ = T[abs(σ)]
    end
    _v = T.(v)
    β = T[10]
    return TrainableDirectionSFKernel{T, typeof(_v)}(k, _σ, β, v)
end
# There are also constructors where the type is infered:
function TrainableDirectionSFKernel(k::Real, Ord::Integer; σ::Union{Nothing, Real} = nothing)
    T = eltype(k)
    return TrainableDirectionSFKernel{T}(k, Ord, σ = σ)
end
function TrainableDirectionSFKernel(k::Real, v::AbstractMatrix{<:Real}; σ::Union{Nothing, Real, AbstractVector{<:Real}} = nothing)
    T = float(eltype(k))
    return TrainableDirectionSFKernel{T}(k, v, σ = σ)
end
function TrainableDirectionSFKernel(k::Real, v::AbstractVector{<:Real}; σ::Union{Nothing, Real} = nothing)
    T = float(eltype(k))
    return TrainableDirectionSFKernel{T}(k, v, σ = σ)
end

@functor TrainableDirectionSFKernel
Flux.@layer TrainableDirectionSFKernel
trainable(a::TrainableDirectionSFKernel) = (; σ = a.σ, β = a.β, v = a.v)

# Unified constructor for the trainable direction spherical Bessel kernels.

function DirectionalSFKernel{T}(k::Real, Ord::Integer; σ::Union{Nothing, Real} = nothing, trainable_direction::Bool = false) where {T<:AbstractFloat}
    if trainable_direction
        return TrainableDirectionSFKernel{T}(k, Ord, σ= σ)
    else
        return FixedDirectionSFKernel{T}(k, Ord, σ = σ)
    end
end
function DirectionalSFKernel{T}(k::T, v::AbstractMatrix{<:Real}; σ::Union{Nothing, Real, AbstractVector{<:Real}} = nothing) where {T<:AbstractFloat}
    if trainable_direction
        return TrainableDirectionSFKernel{T}(k, v, σ= σ)
    else
        return FixedDirectionSFKernel{T}(k, v, σ = σ)
    end
end   
function DirectionalSFKernel{T}(k::T, v::AbstractVector{<:Real}; σ::Union{Nothing, Real} = nothing, trainable_direction::Bool = false) where {T<:AbstractFloat}
    if trainable_direction
        return TrainableDirectionSFKernel{T}(k, v, σ= σ)
    else
        return FixedDirectionSFKernel{T}(k, v, σ = σ)
    end
end
function DirectionalSFKernel(k::Real, Ord::Integer; σ::Union{Nothing, Real} = nothing, trainable_direction::Bool = false)
    T = float(eltype(k))
    return DirectionalSFKernel{T}(k, Ord, σ = σ, trainable_direction = trainable_direction = trainable_direction)
end
function DirectionalSFKernel(k::Real, v::AbstractMatrix{<:Real}; σ::Union{Nothing, Real, AbstractVector{<:Real}} = nothing, trainable_direction::Bool = false)
    T = float(eltype(k))
    return DirectionalSFKernel{T}(k, v, σ = σ, trainable_direction = trainable_direction = trainable_direction)
end
function DirectionalSFKernel(k::Real, v::AbstractVector{<:Real}; σ::Union{Nothing, Real} = nothing, trainable_direction::Bool = false)
    T = float(eltype(k))
    return DirectionalSFKernel{T}(k, v, σ = σ, trainable_direction = trainable_direction = trainable_direction)
end