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
function (a::FixedDirectionPlaneWaveKernel{T})(x::AbstractVector{U}) where {T<:AbstractFloat, U<:Number}
    kx = reshape(sum(a.k*(x .* a.v), dims=1), :)
    i = one(T)im
    return sum(exp.(i*kx) .* a.w)
end

function (a::FixedDirectionPlaneWaveKernel{T})(X::AbstractMatrix{U})::AbstractVector where {T<:AbstractFloat, U<:Number}
    @tullio (+) kx[d, b] := a.v[t, d] * X[t, b]
    i = one(T)im
    return reshape(sum(a.w .* exp.(i*a.k*kx), dims=1), :)
end

function (a::FixedDirectionPlaneWaveKernel{T})(X::AbstractArray{U, 3})::AbstractMatrix where {T<:AbstractFloat, U<:Number}
    @tullio (+) kx[d, b1, b2] := a.v[t, d] * X[t, b1, b2]
    _, B1, B2 = size(X)
    i = one(T)im    
    return reshape(sum(a.w .* exp.(i*a.k*kx), dims=1), (B1, B2))
end

function (a::FixedDirectionPlaneWaveKernel{T})(x1::AbstractVector{U1}, x2::AbstractVector{U2}) where {T<:AbstractFloat, U1<:Number, U2<:Number}
    Δx = reshape(sum(a.k*((x1 - x2) .* a.v), dims=1), :)
    i = one(T)im
    return sum(exp.(i*Δx) .* a.w)
end

function (a::FixedDirectionPlaneWaveKernel{T})(X1::AbstractMatrix{U1}, x2::AbstractVector{U2})::AbstractVector where {T<:AbstractFloat, U1<:Number, U2<:Number}
    X = X1 .-x2 
    @tullio (+) Δx[d, b] := a.v[t, d] * X[t, b]
    i = one(T)im
    return reshape(sum(a.w .* exp.(i*a.k*Δx), dims=1), :)
end

function (a::FixedDirectionPlaneWaveKernel{T})(x1::AbstractVector{U1}, X2::AbstractMatrix{U2})::AbstractVector where {T<:AbstractFloat, U1<:Number, U2<:Number}
    X = x1 .-X2 
    @tullio (+) Δx[d, b] := a.v[t, d] * X[t, b]
    i = one(T)im
    return reshape(sum(a.w .* exp.(i*a.k*Δx), dims=1), :)
end

function (a::FixedDirectionPlaneWaveKernel{T})(X1::AbstractMatrix{U1}, X2::AbstractMatrix{U2})::AbstractMatrix where {T<:AbstractFloat, U1<:Number, U2<:Number}
    @tullio (+) Δx[d, b1, b2] := a.v[t, d] * (X1[t, b1] - X2[t, b2])
    i = one(T)im
    _, B1, B2 = size(Δx)
    return reshape(sum(a.w .* exp.(i*a.k*Δx), dims=1), (B1, B2))
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

function (a::VariableDirectionPlaneWaveKernel{T})(x::AbstractVector{U}) where {T<:AbstractFloat, U<:Number}
    kx = reshape(sum(a.k*(x .* a.v), dims=1), :)
    i = one(T)im
    return sum(exp.(i*kx) .* a.w)
end

function (a::VariableDirectionPlaneWaveKernel{T})(X::AbstractMatrix{U})::AbstractVector where {T<:AbstractFloat, U<:Number}
    @tullio (+) kx[d, b] := a.v[t, d] * X[t, b]
    i = one(T)im
    return reshape(sum(a.w .* exp.(i*a.k*kx), dims=1), :)
end

function (a::VariableDirectionPlaneWaveKernel{T})(X::AbstractArray{U, 3})::AbstractMatrix where {T<:AbstractFloat, U<:Number}
    @tullio (+) kx[d, b1, b2] := a.v[t, d] * X[t, b1, b2]
    i = one(T)im
    _, B1, B2 = size(X)
    return reshape(sum(a.w .* exp.(i*a.k*kx), dims=1), (B1, B2))
end

function (a::VariableDirectionPlaneWaveKernel{T})(x1::AbstractVector{U1}, x2::AbstractVector{U2}) where {T<:AbstractFloat, U1<:Number, U2<:Number}
    Δx = reshape(sum(a.k*((x1 - x2) .* a.v), dims=1), :)
    i = one(T)im
    return sum(exp.(i*Δx) .* a.w)
end

function (a::VariableDirectionPlaneWaveKernel{T})(X1::AbstractMatrix{U1}, x2::AbstractVector{U2})::AbstractVector where {T<:AbstractFloat, U1<:Number, U2<:Number}
    X = X1 .-x2 
    @tullio (+) Δx[d, b] := a.v[t, d] * X[t, b]
    i = one(T)im
    return reshape(sum(a.w .* exp.(i*a.k*Δx), dims=1), :)
end

function (a::VariableDirectionPlaneWaveKernel{T})(x1::AbstractVector{U1}, X2::AbstractMatrix{U2})::AbstractVector where {T<:AbstractFloat, U1<:Number, U2<:Number}
    X = x1 .-X2 
    @tullio (+) Δx[d, b] := a.v[t, d] * X[t, b]
    i = one(T)im
    return reshape(sum(a.w .* exp.(i*a.k*Δx), dims=1), :)
end

function (a::VariableDirectionPlaneWaveKernel{T})(X1::AbstractMatrix{U1}, X2::AbstractMatrix{U2})::AbstractMatrix where {T<:AbstractFloat, U1<:Number, U2<:Number}
    @tullio (+) Δx[d, b1, b2] := a.v[t, d] * (X1[t, b1] - X2[t, b2])
    i = one(T)im
    _, B1, B2 = size(Δx)
    return reshape(sum(a.w .* exp.(i*a.k*Δx), dims=1), (B1, B2))
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

function PlaneWaveKernel(k::T, Ord::N) where {T<:AbstractFloat, U<:Real, N<:Integer}
    return FixedDirectionPlaneWaveKernel{T}(k, Ord)
end

# Plane wave kernel with a neural weight function
# This weight represented by a neural network offers an abstraction to the weight that can be resampled to increase
# complexity without changing the underlying network. 

# We begin by setting a default NN to serve as the weight.
W0 = Chain(Dense(3, 5, tanh),
              Dense(5, 5, tanh),
              Dense(5,3,tanh),
              Dense(3,1,relu))

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

function (a::NeuralWeightPlaneWaveKernel{T, N})(x::AbstractVector{U}) where {T<:AbstractFloat, U<:Number, N<:Integer}
    i = one(T)im
    W = a.W(a.k * a.v)
    sum(exp.(i*a.k*sum(x .* a.v, dims=1)) .* W .* a.w')
end

function (a::NeuralWeightPlaneWaveKernel{T, N})(X::AbstractMatrix{U})::AbstractVector where {T<:AbstractFloat, U<:Number, N<:Integer}
    i = one(T)im
    W = reshape(a.W(a.k * a.v), :)
    @tullio (+) Δx[d, b] := a.v[t, d] * X[t, b]
    return reshape(sum(W.* a.w .* exp.(i*a.k*Δx), dims=1),:)
end

function (a::NeuralWeightPlaneWaveKernel{T, N})(ΔX::AbstractArray{U, 3})::AbstractMatrix where {T<:AbstractFloat, U<:Number, N<:Integer}
    i = one(T)im
    W = reshape(a.W(a.k * a.v), :)
    @tullio (+) Δx[d, b1, b2] := a.v[t, d] * ΔX[t, b1, b2]
    D, B1, B2 = size(Δx)
    return reshape(sum(repeat(reshape(W.* a.w, (D, 1, 1)), 1, B1, B2) .* exp.(i*a.k*Δx), dims=1),(B1, B2))
end

function (a::NeuralWeightPlaneWaveKernel{T, N})(x1::AbstractVector{U1}, x2::AbstractVector{U2}) where {T<:AbstractFloat, U1<:Number, U2<:Number, N<:Integer}
    i = one(T)im
    W = a.W(a.k * a.v)
    sum(exp.(i*a.k*sum((x1 - x2) .* a.v, dims=1)) .* W .* a.w')
end

function (a::NeuralWeightPlaneWaveKernel{T, N})(x1::AbstractVector{U1}, X2::AbstractMatrix{U2})::AbstractVector where {T<:AbstractFloat, U1<:Number, U2<:Number, N<:Integer}
    i = one(T)im
    W = reshape(a.W(a.k * a.v), :)
    ΔX = x1 .- X2
    @tullio (+) Δx[d, b] := a.v[t, d] * ΔX[t, b]
    return reshape(sum(W.* a.w .* exp.(i*a.k*Δx), dims=1),:)
end

function (a::NeuralWeightPlaneWaveKernel{T, N})(X1::AbstractMatrix{U1}, x2::AbstractVector{U2})::AbstractVector where {T<:AbstractFloat, U1<:Number, U2<:Number, N<:Integer}
    i = one(T)im
    W = reshape(a.W(a.k * a.v), :)
    ΔX = X1 .- x2
    @tullio (+) Δx[d, b] := a.v[t, d] * ΔX[t, b]
    return reshape(sum(W.* a.w .* exp.(i*a.k*Δx), dims=1),:)
end

function (a::NeuralWeightPlaneWaveKernel{T, N})(X1::AbstractMatrix{U1}, X2::AbstractMatrix{U2})::AbstractMatrix where {T<:AbstractFloat, U1<:Number, U2<:Number, N<:Integer}
    i = one(T)im
    W = reshape(a.W(a.k * a.v), :)
    @tullio (+) Δx[d, b1, b2] := a.v[t, d] * (X1[t, b1]- X2[t, b2])
    D, B1, B2 = size(Δx)
    return reshape(sum(repeat(reshape(W.* a.w, (D, 1, 1)), 1, B1, B2) .* exp.(i*a.k*Δx), dims=1),(B1, B2))
end

# Flux management

@functor NeuralWeightPlaneWaveKernel
# It is sufficient to inform Julia that the weight function W is the only trainable field and the parameters from W will be extracted directly.
trainable(a::NeuralWeightPlaneWaveKernel) = (; a.W)