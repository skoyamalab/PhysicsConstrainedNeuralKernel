# ====== Source file for the uniform weight kernel implementation ========
# ===== Weight equivalent to considering directionally uniform gain ======

# Struct definition

mutable struct UniformKernel{T} <:ISFKernel{T}
    k::T
end

# Nondefault constructor

function UniformKernel(k::T)::UniformKernel{T} where {T<:AbstractFloat}
    return UniformKernel{T}(T(k))
end

#Implementations of kernel behavior with input

## I have opted to allow for entry of a vector(presumably the difference between inputs) in 2 types of batches in case the operator wants to do anything interesting.

function (a::UniformKernel)(x::AbstractVector{U}) where {U<:Number}
    η = sqrt(sum(x.^2))
    return j0(a.k*η)
end

function (a::UniformKernel)(x::AbstractMatrix{U})::AbstractVector where {U<:Number}
    η = sqrt.(reshape(sum(x.^2, dims=1), :))
    return j0.(a.k*η)
end

function (a::UniformKernel)(x::AbstractArray{U, 3})::AbstractMatrix where {U<:Number}
    _, b1, b2 = size(x)
    η = sqrt.(reshape(sum(x.^2, dims=1), b1, b2))
    return j0.(a.k*η)
end

## Proper kernel entry format, with different implementations depending on inputs.
### Batching format is the same as Flux: vectors are regular inputs, matches are matrices
### where the m-th column represents the m-th input

### Kernel evaluation for 2 vectors
function (a::UniformKernel)(x1::AbstractVector{U1}, x2::AbstractVector{U2}) where {U1<:Number, U2<:Number}
    return j0(a.k*norm(x1-x2))
end

### Kernel evaluation for a batch of inputs and one output
function (a::UniformKernel)(X1::AbstractMatrix{U1}, x2::AbstractVector{U2})::AbstractVector where {U1<:Number, U2<:Number}
    Δx = sqrt.(reshape(sum(abs2, X1 .- x2, dims=1), :))
    return j0.(a.k*Δx)
end

### Kernel evaluation for an input and a batch of inputs
function (a::UniformKernel)(x1::AbstractVector{U1}, X2::AbstractMatrix{U2})::AbstractVector where {U1<:Number, U2<:Number}
    Δx = sqrt.(reshape(sum(abs2, x1 .- X2, dims=1), :))
    return j0.(a.k*Δx)
end

### Kernel evaluation for two batches of inputs
function (a::UniformKernel)(X1::AbstractMatrix{U1}, X2::AbstractMatrix{U2})::AbstractMatrix where {U1<:Number, U2<:Number}
    @tullio (+) ΔX[b1, b2] := (X1[t, b1] - X2[t, b2])^2
    return j0.(a.k*sqrt.(ΔX))
end

# Flux management. Basically, making it so all applications understand how kernels interact with Flux. The actual kernel training is
# mostly package agnostic: the dependencies are ChainRules.jl and Optimisers.jl, which work with mostly arbitrary autodiff backends.
# In theory, this should make the code extremely portable, but Flux is the format I am most familiar with.

## We first make it a functor so that Flux knows to treat this as a learnable function. This kernel has no learnable parameters, but making it a functor means whatever AD backend is chosen will allow derivatives and the like to flow through it.
@functor UniformKernel
# Then we list the learnable parameters (in this case, none). This step is unnecessary here, but it is included just for completeness and parity with other kernels
trainable(a::UniformKernel) = (; )
