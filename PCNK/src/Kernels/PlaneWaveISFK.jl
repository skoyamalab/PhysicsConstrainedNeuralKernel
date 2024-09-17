# ========================= Plane wave kernel formulations ================================
# =========== Kernels based on a plane wave representation of the sound field =============

# Plane wave kernel (non-trainable directions)

abstract type PlaneWaveKernel{T<:AbstractFloat} <: ISFKernel{T} end

# Struct definition

mutable struct FixedDirectionPlaneWaveKernel{T<:AbstractFloat} <: PlaneWaveKernel{T}
    k::T
    w::AbstractArray{<:Number}
    v::AbstractMatrix{T}
end

# Nondefault constructors. In our works, we tend to use the Lebedev grids as default directions. A different choice in direction should use the default constructor.

function FixedDirectionPlaneWaveKernel{T}(k::U, Ord::N)::FixedDirectionPlaneWaveKernel{T} where{T<:AbstractFloat, U<:AbstractFloat, N<:Integer}
    x,y,z,w = lebedev_by_order(Ord)
    v =[x y z]'
    v = Matrix(v ./ sum(abs2, v, dims=1))
    return FixedDirectionPlaneWaveKernel{T}(T(k), T.(w), T.(v))
end

function FixedDirectionPlaneWaveKernel(k::T, Ord::N)::FixedDirectionPlaneWaveKernel{T} where{T<:AbstractFloat, N<:Integer}
    x,y,z,w = lebedev_by_order(Ord)
    v =[x y z]'
    v = Matrix(v ./ sum(abs2, v, dims=1))
    return FixedDirectionPlaneWaveKernel{T}(T(k), T.(w), T.(v))
end

# Implementations
function (a::FixedDirectionPlaneWaveKernel)(x::AbstractVector{<:Number})
    kx = reshape(sum(a.k*(x .* a.v), dims=1), :)
    return sum(exp.(im*kx) .* a.w)
end

function (a::FixedDirectionPlaneWaveKernel)(X::AbstractMatrix{<:Number})::AbstractVector
    @tullio (+) kx[d, b] := a.v[t, d] * X[t, b]
    return reshape(sum(a.w .* exp.(im*a.k*kx), dims=1), :)
end

function (a::FixedDirectionPlaneWaveKernel)(X::AbstractArray{<:Number, 3})::AbstractMatrix
    @tullio (+) kx[d, b1, b2] := a.v[t, d] * X[t, b1, b2]
    _, B1, B2 = size(X)
    return reshape(sum(a.w .* exp.(im*a.k*kx), dims=1), (B1, B2))
end

function (a::FixedDirectionPlaneWaveKernel)(x1::AbstractVector{<:Number}, x2::AbstractVector{<:Number})
    Δx = reshape(sum(a.k*((x1 - x2) .* a.v), dims=1), :)
    return sum(exp.(im*Δx) .* a.w)
end

function (a::FixedDirectionPlaneWaveKernel)(X1::AbstractMatrix{<:Number}, x2::AbstractVector{<:Number})::AbstractVector
    X = X1 .-x2 
    @tullio (+) Δx[d, b] := a.v[t, d] * X[t, b]
    return reshape(sum(a.w .* exp.(im*a.k*Δx), dims=1), :)
end

function (a::FixedDirectionPlaneWaveKernel)(x1::AbstractVector{<:Number}, X2::AbstractMatrix{<:Number})::AbstractVector
    X = x1 .-X2 
    @tullio (+) Δx[d, b] := a.v[t, d] * X[t, b]
    return reshape(sum(a.w .* exp.(im*a.k*Δx), dims=1), :)
end

function (a::FixedDirectionPlaneWaveKernel)(X1::AbstractMatrix{<:Number}, X2::AbstractMatrix{<:Number})::AbstractMatrix
    @tullio (+) Δx[d, b1, b2] := a.v[t, d] * (X1[t, b1] - X2[t, b2])
    _, B1, B2 = size(Δx)
    return reshape(sum(a.w .* exp.(im*a.k*Δx), dims=1), (B1, B2))
end

#Flux management

@functor FixedDirectionPlaneWaveKernel
# The only limiting parameter is the combining weight for each direction, so we must inform Flux of that.
trainable(a::FixedDirectionPlaneWaveKernel) = (; a.w)

# Plane wave kernel (trainable directions)
# This kernel, frankly, is not very useful. The directions must remain unit norm to respect the Helmholtz equation which adds unnecessary restrictions to the training that can be circumvented by applying a sparsity criterion on regularly-sampled fixed directions.
#This kernel is still included for completeness, but initializing a plane wave kernel defaults to the fixed direction version.


# Definition of struct

mutable struct VariableDirectionPlaneWaveKernel{T<:AbstractFloat} <:PlaneWaveKernel{T}
    k::T
    w::AbstractArray{<:Number}
    v::AbstractMatrix{<:Number}
end

# Nondefault constructors

function VariableDirectionPlaneWaveKernel{T}(k::U, Ord::N)::VariableDirectionPlaneWaveKernel{T} where{T<:AbstractFloat, U<:AbstractFloat, N<:Integer}
    x,y,z,w = lebedev_by_order(Ord)
    v =[x y z]'
    v = Matrix(v ./ sum(abs2, v, dims=1))
    return VariableDirectionPlaneWaveKernel{T}(T(k), T.(w), T.(v))
end

function VariableDirectionPlaneWaveKernel(k::T, Ord::N)::VariableDirectionPlaneWaveKernel{T} where{T<:AbstractFloat, N<:Integer}
    x,y,z,w = lebedev_by_order(Ord)
    v =[x y z]'
    v = Matrix(v ./ sum(abs2, v, dims=1))
    return VariableDirectionPlaneWaveKernel{T}(T(k), T.(w), T.(v))
end



#Implementations

function (a::VariableDirectionPlaneWaveKernel)(x::AbstractVector{<:Number})
    kx = reshape(sum(a.k*(x .* a.v), dims=1), :)
    return sum(exp.(im*kx) .* a.w)
end

function (a::VariableDirectionPlaneWaveKernel)(X::AbstractMatrix{<:Number})::AbstractVector
    @tullio (+) kx[d, b] := a.v[t, d] * X[t, b]
    return reshape(sum(a.w .* exp.(im*a.k*kx), dims=1), :)
end

function (a::VariableDirectionPlaneWaveKernel)(X::AbstractArray{<:Number, 3})::AbstractMatrix
    @tullio (+) kx[d, b1, b2] := a.v[t, d] * X[t, b1, b2]
    _, B1, B2 = size(X)
    return reshape(sum(a.w .* exp.(im*a.k*kx), dims=1), (B1, B2))
end

function (a::VariableDirectionPlaneWaveKernel)(x1::AbstractVector{<:Number}, x2::AbstractVector{<:Number})
    Δx = reshape(sum(a.k*((x1 - x2) .* a.v), dims=1), :)
    return sum(exp.(im*Δx) .* a.w)
end

function (a::VariableDirectionPlaneWaveKernel)(X1::AbstractMatrix{<:Number}, x2::AbstractVector{<:Number})::AbstractVector
    X = X1 .-x2 
    @tullio (+) Δx[d, b] := a.v[t, d] * X[t, b]
    return reshape(sum(a.w .* exp.(im*a.k*Δx), dims=1), :)
end

function (a::VariableDirectionPlaneWaveKernel)(x1::AbstractVector{<:Number}, X2::AbstractMatrix{<:Number})::AbstractVector
    X = x1 .-X2 
    @tullio (+) Δx[d, b] := a.v[t, d] * X[t, b]
    return reshape(sum(a.w .* exp.(im*a.k*Δx), dims=1), :)
end

function (a::VariableDirectionPlaneWaveKernel)(X1::AbstractMatrix{<:Number}, X2::AbstractMatrix{<:Number})::AbstractMatrix
    @tullio (+) Δx[d, b1, b2] := a.v[t, d] * (X1[t, b1] - X2[t, b2])
    _, B1, B2 = size(Δx)
    return reshape(sum(a.w .* exp.(im*a.k*Δx), dims=1), (B1, B2))
end

# Flux management

@functor VariableDirectionPlaneWaveKernel

# Converters
#It is possible to convert between kernels, in case training is to be performed in separate stages or other similar unique use cases.

function FixedDirectionPlaneWaveKernel{T}(a::VariableDirectionPlaneWaveKernel{U})::FixedDirectionPlaneWaveKernel{T} where {T<:AbstractFloat, U<:AbstractFloat}
    return FixedDirectionPlaneWaveKernel{T}(T(a.k), T.(a.w), T.(a.v))
end

function FixedDirectionPlaneWaveKernel(a::VariableDirectionPlaneWaveKernel{T})::FixedDirectionPlaneWaveKernel{T} where {T<:AbstractFloat}
    return FixedDirectionPlaneWaveKernel{T}(T(a.k), T.(a.w), T.(a.v))
end

function VariableDirectionPlaneWaveKernel{T}(a::FixedDirectionPlaneWaveKernel{U})::VariableDirectionPlaneWaveKernel{T} where {T<:AbstractFloat, U<:AbstractFloat}
    return VariableDirectionPlaneWaveKernel{T}(T(a.k), T.(a.w), T.(a.v))
end

function VariableDirectionPlaneWaveKernel(a::FixedDirectionPlaneWaveKernel{T})::VariableDirectionPlaneWaveKernel{T} where {T<:AbstractFloat}
    return VariableDirectionPlaneWaveKernel{T}(T(a.k), T.(a.w), T.(a.v))
end

# As mentioned previously, we also include a default constructor to the entire Plane Wave Kernel framework.

function PlaneWaveKernel{T}(k::U, w::AbstractVector{<:Real}, v::AbstractMatrix{<:Real}) where {T<:AbstractFloat, U<:Real}
    return FixedDirectionPlaneWaveKernel{T}(T(k), T.(w),T.(v))
end

function PlaneWaveKernel(k::T, w::AbstractVector{<:Real}, v::AbstractMatrix{<:Real}) where {T<:AbstractFloat}
    return FixedDirectionPlaneWaveKernel{T}(k, T.(w),T.(v))
end

function PlaneWaveKernel{T}(k::U, Ord::N) where {T<:AbstractFloat, U<:Real, N<:Integer}
    return FixedDirectionPlaneWaveKernel{T}(k,Ord)
end

function PlaneWaveKernel(k::T, Ord::N) where {T<:AbstractFloat, N<:Integer}
    return FixedDirectionPlaneWaveKernel{T}(k, Ord)
end

# Plane wave kernel with a neural weight function
# This weight represented by a neural network offers an abstraction to the weight that can be resampled to increase
# complexity without changing the underlying network. 

# We begin by setting a default NN to serve as the weight.
# This is the NN in the paper, which is costly and required multiple starting points.
# dW0 = NODE(Dense(8, 8, tanh))
# W0 = Chain(Dense(3,8, tanh),
#           dW0,
#           Dense(8,5,tanh),
#           Dense(5, 1),
#           x->relu.(tanh.(x)))
# The default we will use is more compact and simpler in order for the training to reach a feasible conclusion in a more stable fashion.
W0 = Chain(Dense(3,2, tanh),
           Dense(2,1,tanh),
           x->relu.(x))

#Now we can define the struct.

mutable struct NeuralWeightPlaneWaveKernel{T<:AbstractFloat, N<:Integer} <:NeuralISFKernel{T, N}
    k::T
    Ord::N
    W
    v::AbstractMatrix{T}
    w::AbstractVector{T}
end

# The NN is not given any type as to not restrict what kind of weight is used. The surrounding ecosystem is very capable of infering how to let derivatives flow through whatever construct is placed, as well as performing conversions to the GPU and the like.

# Constructors
# The kernel can be constructed by giving the wavenumber and an order for the integrator.
# The weight function NN (the chain) can be optionally supplied.
# Note that the struct is mutable. You can change the NN if you want.

function NeuralWeightPlaneWaveKernel{T, N}(k::U, Ord::N2)::NeuralWeightPlaneWaveKernel{T,N} where {T<:AbstractFloat, U<:AbstractFloat, N<:Integer, N2<:Integer}
    W = W0
    x,y,z,w = lebedev_by_order(Ord)
    v = Matrix{T}([x y z]')
    return NeuralWeightPlaneWaveKernel{T,N}(T(k), N(Ord), W, v, T.(w))
end

function NeuralWeightPlaneWaveKernel{T}(k::U, Ord::N)::NeuralWeightPlaneWaveKernel{T,N} where {T<:AbstractFloat, U<:AbstractFloat, N<:Integer}
    W = W0
    x,y,z,w = lebedev_by_order(Ord)
    v = Matrix{T}([x y z]')
    return NeuralWeightPlaneWaveKernel{T,N}(T(k), Ord, W, v, T.(w))
end

function NeuralWeightPlaneWaveKernel{N}(k::T, Ord::N2)::NeuralWeightPlaneWaveKernel{T,N} where {T<:AbstractFloat, N<:Integer, N2<:Integer}
    W = W0
    x,y,z,w = lebedev_by_order(Ord)
    v = Matrix{T}([x y z]')
    return NeuralWeightPlaneWaveKernel{T,N}(k, N(Ord), W, v, T.(w))
end

function NeuralWeightPlaneWaveKernel(k::T, Ord::N)::NeuralWeightPlaneWaveKernel{T,N} where {T<:AbstractFloat, N<:Integer}
    W = W0
    x,y,z,w = lebedev_by_order(Ord)
    v = Matrix{T}([x y z]')
    return NeuralWeightPlaneWaveKernel{T,N}(k, Ord, W, v, T.(w))
end

function NeuralWeightPlaneWaveKernel{T, N}(k::U, Ord::N2, W)::NeuralWeightPlaneWaveKernel{T,N} where {T<:AbstractFloat, U<:AbstractFloat, N<:Integer, N2<:Integer}
    x,y,z,w = lebedev_by_order(Ord)
    v = Matrix{T}([x y z]')
    return NeuralWeightPlaneWaveKernel{T,N}(T(k), N(Ord), W, v, T.(w))
end

function NeuralWeightPlaneWaveKernel{T}(k::U, Ord::N, W)::NeuralWeightPlaneWaveKernel{T,N} where {T<:AbstractFloat, U<:AbstractFloat, N<:Integer}
    x,y,z,w = lebedev_by_order(Ord)
    v = Matrix{T}([x y z]')
    return NeuralWeightPlaneWaveKernel{T,N}(T(k), Ord, W, v, T.(w))
end

function NeuralWeightPlaneWaveKernel{N}(k::T, Ord::N2, W)::NeuralWeightPlaneWaveKernel{T,N} where {T<:AbstractFloat, N<:Integer, N2<:Integer}
    x,y,z,w = lebedev_by_order(Ord)
    v = Matrix{T}([x y z]')
    return NeuralWeightPlaneWaveKernel{T,N}(k, N(Ord), W, v, T.(w))
end

function NeuralWeightPlaneWaveKernel(k::T, Ord::N, W)::NeuralWeightPlaneWaveKernel{T,N} where {T<:AbstractFloat, N<:Integer}
    x,y,z,w = lebedev_by_order(Ord)
    v = Matrix{T}([x y z]')
    return NeuralWeightPlaneWaveKernel{T,N}(k, Ord, W, v, T.(w))
end

# Implementations

function (a::NeuralWeightPlaneWaveKernel)(x::AbstractVector{<:Number})
    W = a.W(a.k * a.v)
    sum(exp.(im*a.k*sum(x .* a.v, dims=1)) .* W .* a.w')
end

function (a::NeuralWeightPlaneWaveKernel)(X::AbstractMatrix{<:Number})::AbstractVector
    W = reshape(a.W(a.k * a.v), :)
    @tullio (+) Δx[d, b] := a.v[t, d] * X[t, b]
    return reshape(sum(W.* a.w .* exp.(im*a.k*Δx), dims=1),:)
end

function (a::NeuralWeightPlaneWaveKernel)(ΔX::AbstractArray{<:Number, 3})::AbstractMatrix
    W = reshape(a.W(a.k * a.v), :)
    @tullio (+) Δx[d, b1, b2] := a.v[t, d] * ΔX[t, b1, b2]
    D, B1, B2 = size(Δx)
    return reshape(sum(repeat(reshape(W.* a.w, (D, 1, 1)), 1, B1, B2) .* exp.(im*a.k*Δx), dims=1),(B1, B2))
end

function (a::NeuralWeightPlaneWaveKernel)(x1::AbstractVector{<:Number}, x2::AbstractVector{<:Number})
    W = a.W(a.k * a.v)
    sum(exp.(im*a.k*sum((x1 - x2) .* a.v, dims=1)) .* W .* a.w')
end

function (a::NeuralWeightPlaneWaveKernel)(x1::AbstractVector{<:Number}, X2::AbstractMatrix{<:Number})::AbstractVector
    W = reshape(a.W(a.k * a.v), :)
    ΔX = x1 .- X2
    @tullio (+) Δx[d, b] := a.v[t, d] * ΔX[t, b]
    return reshape(sum(W.* a.w .* exp.(im*a.k*Δx), dims=1),:)
end

function (a::NeuralWeightPlaneWaveKernel)(X1::AbstractMatrix{<:Number}, x2::AbstractVector{<:Number})::AbstractVector
    W = reshape(a.W(a.k * a.v), :)
    ΔX = X1 .- x2
    @tullio (+) Δx[d, b] := a.v[t, d] * ΔX[t, b]
    return reshape(sum(W.* a.w .* exp.(im*a.k*Δx), dims=1),:)
end

function (a::NeuralWeightPlaneWaveKernel)(X1::AbstractMatrix{<:Number}, X2::AbstractMatrix{<:Number})::AbstractMatrix
    W = reshape(a.W(a.k * a.v), :)
    @tullio (+) Δx[d, b1, b2] := a.v[t, d] * (X1[t, b1]- X2[t, b2])
    D, B1, B2 = size(Δx)
    return reshape(sum(repeat(reshape(W.* a.w, (D, 1, 1)), 1, B1, B2) .* exp.(im*a.k*Δx), dims=1),(B1, B2))
end

# Flux management

@functor NeuralWeightPlaneWaveKernel
# It is sufficient to inform Julia that the weight function W is the only trainable field and the parameters from W will be extracted directly.
trainable(a::NeuralWeightPlaneWaveKernel) = (; a.W)