# Generic object for the plane wave kernels

abstract type PlaneWaveKernel{T<:AbstractFloat} <: ISFKernel{T} end
abstract type DiscreteWeightPlaneWaveKernel{T<:AbstractFloat} <:PlaneWaveKernel{T} end
# Struct definition of the plane wave kernels.

mutable struct FixedDirectionPlaneWaveKernel{T<:AbstractFloat} <: DiscreteWeightPlaneWaveKernel{T}
    k::T
    σ::AbstractArray
    v::AbstractMatrix{T}
end

# Non-generic constructors
function FixedDirectionPlaneWaveKernel{T}(k::Real, Ord::Integer; σ::Union{Nothing, Real} = nothing) where {T<:AbstractFloat}
    x,y,z, w = lebedev_by_order(Ord)
    v =[x y z]'
    v = Matrix{T}(v ./ sum(abs2, v, dims=1))
    if isnothing(σ)
        _σ = T.(w)
    else
        _σ = T.(abs(σ) * w)
    end
    return FixedDirectionPlaneWaveKernel(T(k), _σ, v)
end
function FixedDirectionPlaneWaveKernel{T}(k::Real, v::AbstractMatrix{<:Real}; σ::Union{Nothing, Real, AbstractVector{<:Real}} = nothing) where {T<:AbstractFloat}
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
    return FixedDirectionPlaneWaveKernel(T(k), _σ, T.(v))
end
function FixedDirectionPlaneWaveKernel(k::U, Ord::Integer; σ::Union{Nothing, Real} = nothing) where {U<:Real}
    T = float(U)
    return FixedDirectionPlaneWaveKernel{T}(k, Ord, σ = σ)
end
function FixedDirectionPlaneWaveKernel(k::U, v::AbstractMatrix{<:Real}; σ::Union{Nothing, Real, AbstractVector{<:Real}} = nothing) where {U<:Real}
    T = float(U)
    return FixedDirectionPlaneWaveKernel{T}(k, v, σ = σ)
end

@functor FixedDirectionPlaneWaveKernel
Flux.@layer FixedDirectionPlaneWaveKernel
trainable(a::FixedDirectionPlaneWaveKernel) = (; σ = a.σ,)

# Plane wave kernels with trainable directions.

mutable struct TrainableDirectionPlaneWaveKernel{T<:AbstractFloat} <: DiscreteWeightPlaneWaveKernel{T}
    k::T
    σ::AbstractVector
    v::AbstractMatrix
end

# Non-generic constructors
function TrainableDirectionPlaneWaveKernel{T}(k::Real, Ord::Integer; σ::Union{Nothing, Real} = nothing) where {T<:AbstractFloat}
    x,y,z, w = lebedev_by_order(Ord)
    v =[x y z]'
    v = Matrix{T}(v ./ sum(abs2, v, dims=1))
    if isnothing(σ)
        _σ = T.(w)
    else
        _σ = T.(abs(σ) * w)
    end
    return TrainableDirectionPlaneWaveKernel(T(k), _σ, v)
end
function TrainableDirectionPlaneWaveKernel{T}(k::Real, v::AbstractMatrix{<:Real}; σ::Union{Nothing, Real, AbstractVector{<:Real}} = nothing) where {T<:AbstractFloat}
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
    return TrainableDirectionPlaneWaveKernel(T(k), _σ, T.(v))
end
function TrainableDirectionPlaneWaveKernel(k::U, Ord::Integer; σ::Union{Nothing, Real} = nothing) where {U<:Real}
    T = float(U)
    return TrainableDirectionPlaneWaveKernel{T}(k, Ord, σ = σ)
end
function TrainableDirectionPlaneWaveKernel(k::U, v::AbstractMatrix{<:Real}; σ::Union{Nothing, Real, AbstractVector{<:Real}} = nothing) where {U<:Real}
    T = float(U)
    return TrainableDirectionPlaneWaveKernel{T}(k, v, σ = σ)
end



@functor TrainableDirectionPlaneWaveKernel
Flux.@layer TrainableDirectionPlaneWaveKernel
trainable(a::TrainableDirectionPlaneWaveKernel) = (; σ = a.σ, v = a.v)

# Plane wave kernel with a neural weight.

## Default weight function
# We begin by setting a default NN to serve as the weight.
# This is the NN in the paper, which is costly and required multiple starting points.
# __PWISFK_dW0 = NODE(Dense(8, 8, tanh))
# __PWFISFK_W0 = Chain(Dense(3,8, tanh),
#           __PWISFK_dW0,
#           Dense(8,5,tanh),
#           Dense(5, 1),
#           x->relu.(tanh.(x)))
# The default we will use is more compact and simpler in order for the training to reach a feasible conclusion in a more stable fashion.
__PWISFK_W0 = Chain(Dense(3,2, tanh),
           Dense(2,1,tanh),
           softplus)

mutable struct NeuralWeightPlaneWaveKernel{T<:AbstractFloat} <: PlaneWaveKernel{T}
    k::T
    W::Chain
    σ::AbstractVector{T}
    v::AbstractMatrix{T}
end

# Non-generic constructors
function NeuralWeightPlaneWaveKernel{T}(k::Real, D::NamedTuple, σ::AbstractVector{<:Real}, v::AbstractMatrix{<:Real}) where {T<:AbstractFloat}
    return NeuralWeightPlaneWaveKernel(T(k), Chain(D ...), σ, v)
end
function NeuralWeightPlaneWaveKernel{T}(k::Real, Ord::Integer; σ::Union{Nothing, Real} = nothing, W = __PWISFK_W0) where {T<:AbstractFloat}
    x,y,z, w = lebedev_by_order(Ord)
    v =[x y z]'
    v = Matrix{T}(v ./ sum(abs2, v, dims=1))
    if isnothing(σ)
        _σ = T.(w)
    else
        _σ = T.(abs(σ) * w)
    end
    return NeuralWeightPlaneWaveKernel{T}(T(k), T(W), _σ, v)
end
function NeuralWeightPlaneWaveKernel{T}(k::Real, v::AbstractMatrix{<:Real}; σ::Union{Nothing, Real, AbstractVector{<:Real}} = nothing, W = __PWISFK_W0) where {T<:AbstractFloat}
    N = size(v, 2)
    if isnothing(σ)
        _σ = T.(ones(N)/N)
    elseif isa(σ, Real)
        _σ = T.(abs(σ) * ones(N)/N)
    else
        if length(σ) != N
            error("The number of weights has to correspond to the number of directions!")
        else
            _σ = T.(abs.(σ))
        end
    end
    return NeuralWeightPlaneWaveKernel{T}(T(k), T(W), _σ, T.(v))
end
function NeuralWeightPlaneWaveKernel(k::U, D::NamedTuple, σ::AbstractVector{<:Real}, v::AbstractMatrix{<:Real}) where {U<:Real}
    T = float(U)
    return NeuralWeightPlaneWaveKernel{T}(k, D, σ, v)
end
function NeuralWeightPlaneWaveKernel(k::U, Ord::Integer; σ::Union{Nothing, Real} = nothing, W = __PWISFK_W0) where {U<:Real}
    return NeuralWeightPlaneWaveKernel{float(U)}(k, Ord; σ = σ, W = W)
end
function NeuralWeightPlaneWaveKernel(k::U, v::AbstractMatrix{<:Real}; σ::Union{Nothing, Real, AbstractVector{<:Real}} = nothing, W = __PWISFK_W0) where {U<:Real}
    return NeuralWeightPlaneWaveKernel{float(U)}(k, v; σ = σ, W = W)
end

@functor NeuralWeightPlaneWaveKernel
Flux.@layer NeuralWeightPlaneWaveKernel
trainable(a::NeuralWeightPlaneWaveKernel) = (;W=trainable(a.W))

# Generic constructor that combines the definitions in here.

function PlaneWaveKernel{T}(k::Real, Ord::Integer; σ::Union{Nothing, Real} = nothing, W = nothing, trainable_direction::Bool = false) where {T<:AbstractFloat}
    if isnothing(W) && trainable_direction
        return TrainableDirectionPlaneWaveKernel{T}(k, Ord, σ = σ)
    elseif isnothing(W) && !trainable_direction
        return FixedDirectionPlaneWaveKernel{T}(k, Ord, σ = σ)
    elseif !isnothing(W) && !trainable_direction
        return NeuralWeightPlaneWaveKernel{T}(k, Ord, σ = σ, W = W)
    else
        error("Neural weight kernel does not admit trainable directions! The grid used for integration can be refactored if changes are needed!")
    end
end

function PlaneWaveKernel{T}(k::Real, v::AbstractMatrix{<:Real}; σ::Union{Nothing, Real} = nothing, W = nothing, trainable_direction::Bool = false) where {T<:AbstractFloat}
    if isnothing(W) && trainable_direction
        return TrainableDirectionPlaneWaveKernel{T}(k, v, σ = σ)
    elseif isnothing(W) && !trainable_direction
        return FixedDirectionPlaneWaveKernel{T}(k, v, σ = σ)
    elseif !isnothing(W) && !trainable_direction
        return NeuralWeightPlaneWaveKernel{T}(k, v, σ = σ, W = W)
    else
        error("Neural weight kernel does not admit trainable directions! The grid used for integration can be refactored if changes are needed!")
    end
end

function PlaneWaveKernel(k::U, Ord::Integer; σ::Union{Nothing, Real} = nothing, W = nothing, trainable_direction::Bool = false) where {U<:Real}
    T = float(U)
    return PlaneWaveKernel{T}(k, Ord, σ = σ, W = W, trainable_direction = trainable_direction)
end

function PlaneWaveKernel(k::U, v::AbstractMatrix{<:Real}; σ::Union{Nothing, Real} = nothing, W = nothing, trainable_direction::Bool = false) where {U<:Real}
    T = float(U)
    return PlaneWaveKernel{T}(k, v, σ = σ, W = W, trainable_direction = trainable_direction)
end