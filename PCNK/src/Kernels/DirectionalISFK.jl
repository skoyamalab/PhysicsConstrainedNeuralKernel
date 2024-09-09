# ====== Source file for the directional weight kernel implementations ========
# ===== Weight equivalent to biasing estimation towards 1 direction ======

# This kernel function is part of a family of infinite-dimensional kernel functions that bias the estimation towards a single direction.

abstract type DirectionalKernel{T<:AbstractFloat} <:ISFKernel{T} end

# Simple directional kernel (non-trainable direction)

# Struct definition

mutable struct FixedDirectionSFKernel{T<:AbstractFloat} <:DirectionalKernel{T}
    k::T
    β::AbstractVector{<:Number}
    v::AbstractVector{T}
end

# Constructors

# You inform the constructor of the vector you want to serve as the directional bias.
# The constructor infers the type and then assigns a random value for β.

function FixedDirectionSFKernel{T1}(k::T2, V::AbstractVector{T3})::FixedDirectionSFKernel{T1} where {T1<:AbstractFloat, T2<:Real, T3<:Real}
    v = T1.(V/norm(V))
    β = 10*rand(T1, 1)
    return FixedDirectionSFKernel{T1}(T1(k), β, v)
end

function FixedDirectionSFKernel(k::T1, V::AbstractVector{T2})::FixedDirectionSFKernel{T1} where {T1<:AbstractFloat, T2<:Real}
    v = T1.(V/norm(V))
    β = 10*rand(T1, 1)
    return FixedDirectionSFKernel{T1}(k, β, v)
end

function FixedDirectionSFKernel(k::T2, V::AbstractVector{T1})::FixedDirectionSFKernel{T1} where {T1<:AbstractFloat, T2<:Integer}
    v = T1.(V/norm(V))
    β = 10*rand(T1, 1)
    return FixedDirectionSFKernel{T1}(T1(k), β, v)
end

# Implementations

## Same as the uniform weight kernel, the directional kernel also changes behavior depending on input.

## In case a single input is given, the kernel considers it so you calculated the difference yourself.

function (a::FixedDirectionSFKernel{T})(x::AbstractVector{T})::Complex{T} where {T<:AbstractFloat}
    i = one(T)im
    β = view(a.β, 1)
    return j0.(sqrt(sum((Complex.(a.k*x) - i*(β .*a.v)).^2)))/i0.(β)
end

function (a::FixedDirectionSFKernel{T})(x::AbstractMatrix{T})::AbstractVector{Complex{T}} where {T<:AbstractFloat}
    i = one(T)im
    β = view(a.β, 1)
    return j0.(sqrt.(reshape(sum((Complex.(a.k*x) .- i*(β .*a.v)).^2, dims=1), :)))/i0.(β)
end

function (a::FixedDirectionSFKernel{T})(ΔX::AbstractArray{T, 3})::AbstractMatrix{Complex{T}} where {T<:AbstractFloat}#Format for input is coordinates, vertical batch and horizontal batch
    i = one(T)im
    β = view(a.β, 1)
    d, B1, B2 = size(ΔX)
    return j0.(sqrt.(reshape(sum((a.k*ΔX  - i*repeat(reshape(β .*a.v, (d, 1, 1)), 1, B1, B2)).^2, dims=1), B1, B2)))/i0.(β)
end

## If two inputs are given, the behavior is more similar to how a kernel is meant to work.
## Even though the packages are meant to be agnostic to the ML package or autodiff backend,
## we still considered the Flux format of input vectors and matrix batches

function (a::FixedDirectionSFKernel{T})(x1::AbstractVector{T}, x2::AbstractVector{T})::Complex{T} where {T<:AbstractFloat}
    i = one(T)im
    Δx = Complex.(a.k*(x1 - x2))
    β = view(a.β, 1)
    return j0(sqrt(sum((Δx - i*(β .*a.v)).^2)))/i0.(β)
end

function (a::FixedDirectionSFKernel{T})(X1::AbstractMatrix{T}, x2::AbstractVector{T})::AbstractVector{Complex{T}} where {T<:AbstractFloat}
    i = one(T)im
    Δx = X1 .- x2
    return j0.(sqrt.(reshape(sum((Complex.(a.k*Δx) .- i*(a.β .*a.v)).^2, dims=1), :)))./i0.(a.β)
end

function (a::FixedDirectionSFKernel{T})(x1::AbstractVector{T}, X2::AbstractMatrix{T})::AbstractVector{Complex{T}} where {T<:AbstractFloat}
    i = one(T)im
    Δx = x1 .- X2
    return j0.(sqrt.(reshape(sum((Complex.(a.k*Δx) .- i*(a.β .*a.v)).^2, dims=1), :)))./i0.(a.β)
end

function (a::FixedDirectionSFKernel{T})(X1::AbstractMatrix{T}, X2::AbstractMatrix{T})::AbstractMatrix{Complex{T}} where {T<:AbstractFloat}
    i = one(T)im
    @tullio Δx[t, b1, b2] := X1[t, b1] - X2[t, b2]
    D, B1, B2 = size(Δx)
    ΔX = sqrt.(reshape(sum((Complex.(a.k*Δx) - i*repeat(reshape(a.β .*a.v, (D,1,1)), 1, B1, B2)).^2, dims=1), B1, B2))
    return j0.(ΔX)./i0.(a.β)
end

# Flux management. The only trainable parameter is β, and the direction is meant to be fixed.
# Usually β is the source direction. But, for the sake of completeness this kernel is offered as an alternative.

@functor FixedDirectionSFKernel
trainable(a::FixedDirectionSFKernel) = (; a.β)

# Simple directional kernel (trainable direction)
# In this case,the direction is considered unknown and instead we have a bias vector β that will supply both magnitude and direction/

mutable struct VariableDirectionSFKernel{T<:AbstractFloat} <:DirectionalKernel{T}
    k::T
    β::AbstractVector{<:Number}
end

# Non-default constructors that basically only take the wave number now since everything about β is learnable.

function VariableDirectionSFKernel{T}(k::Real)::VariableDirectionSFKernel{T} where {T<:AbstractFloat}
    β = 10*randn(T1, 3) .+ one(T)
    return VariableDirectionSFKernel{T}(T1(k), β)
end

function VariableDirectionSFKernel(k::T)::VariableDirectionSFKernel{T} where {T<:AbstractFloat}
    β = 10*randn(T, 3) .+ one(T)
    return VariableDirectionSFKernel{T}(k, β)
end

# Implementations considering the various possible inputs

function (a::VariableDirectionSFKernel{T})(x::AbstractVector{U}) where {T<:AbstractFloat, U<:Number}
    i = one(T)im
    return j0(sqrt(sum((a.k*x - i*a.β).^2)))/i0(norm(a.β))
end

function (a::VariableDirectionSFKernel{T})(X::AbstractMatrix{U})::AbstractVector where {T<:AbstractFloat, U<:Number}
    i = one(T)im
    return j0.(sqrt.(reshape(sum((a.k*X .- i*a.β).^2, dims=1), :)))/i0(norm(a.β))
end

function (a::VariableDirectionSFKernel{T})(ΔX::AbstractArray{U, 3})::AbstractMatrix where {T<:AbstractFloat, U<:Number}
    i = one(T)im
    d, B1, B2 = size(ΔX)
    return j0.(sqrt.(reshape(sum((a.k*ΔX - i*repeat(reshape(a.β, (d, 1, 1)), 1, B1, B2)).^2, dims=1), B1, B2)))/i0(norm(a.β))
end

function (a::VariableDirectionSFKernel{T})(x1::AbstractVector{U1}, x2::AbstractVector{U2}) where {T<:AbstractFloat, U1<:Number, U2<:Number}
    i = one(T)im
    Δx = Complex.(a.k*(x1 - x2))
    return j0(sqrt(sum((Δx - i*a.β).^2)))/i0(norm(a.β))
end

function (a::VariableDirectionSFKernel{T})(X1::AbstractMatrix{U1}, x2::AbstractVector{U2})::AbstractVector where {T<:AbstractFloat, U1<:Number, U2<:Number}
    i = one(T)im
    Δx = Complex.(a.k*(X1 .- x2))
    return j0.(sqrt.(reshape(sum((Δx .- i*a.β).^2, dims=1), :)))/i0(norm(a.β))
end

function (a::VariableDirectionSFKernel{T})(x1::AbstractVector{U1}, X2::AbstractMatrix{U2})::AbstractVector where {T<:AbstractFloat, U1<:Number, U2<:Number}
    i = one(T)im
    Δx = Complex.(a.k*(x1 .- X2))
    return j0.(sqrt.(reshape(sum((Δx .- i*a.β).^2, dims=1), :)))/i0(norm(a.β))
end

function (a::VariableDirectionSFKernel{T})(X1::AbstractMatrix{U1}, X2::AbstractMatrix{U2})::AbstractMatrix where {T<:AbstractFloat, U1<:Number, U2<:Number}
    i = one(T)im
    @tullio Δx[t,b1,b2] := X1[t, b1] - X2[t, b2]
    D, B1, B2 = size(Δx)
    ΔX = sqrt.(reshape(sum((Complex.(a.k*Δx) - i*repeat(reshape(a.β, (D,1,1)), 1, B1, B2)).^2, dims=1), (B1, B2)))
    return j0.(ΔX)/i0(norm(a.β))
end

# Flux management to make it trainable. It is unnecessary to list β as a parameter since Flux only considers vector input.
# Therefore, just by making it a functor, Flux would already assume only β is trainable.
# However, since the ML platform used might not be Flux, I list it in order to error in the side of caution.
# Also, if one day Flux changes and starts allowing scalar parameters the code will still work.

@functor VariableDirectionSFKernel
trainable(a::VariableDirectionSFKernel) = (; a.β)

# Converters that allow for turning a variable direction kernel into a fixed direction one and vice-versa.

function FixedDirectionSFKernel{T}(a::VariableDirectionSFKernel{U})::FixedDirectionSFKernel{T} where {T<:AbstractFloat, U<:AbstractFloat}
    β = T.(similar(a.β, 1))
    copyto!(η, norm(a.β))
    v = a.β/norm(a.β)
    return FixedDirectionSFKernel{T}(a.k, β, v)
end

function FixedDirectionSFKernel(a::VariableDirectionSFKernel{T})::FixedDirectionSFKernel{T} where {T<:AbstractFloat}
    β = T.(similar(a.β, 1))
    copyto!(η, norm(a.β))
    v = a.β/norm(a.β)
    return FixedDirectionSFKernel{T}(a.k, β, v)
end

function VariableDirectionSFKernel{T}(a::FixedDirectionSFKernel{U})::VariableDirectionSFKernel{T} where {T<:AbstractFloat, U<:AbstractFloat}
    return VariableDirectionSFKernel{T}(T(a.k), T.(a.β.*a.v) )
end

function VariableDirectionSFKernel(a::FixedDirectionSFKernel{T})::VariableDirectionSFKernel{T} where {T<:AbstractFloat}
    return VariableDirectionSFKernel{T}(T(a.k), T.(a.β.*a.v) )
end

# Constructor for the overall directionally-biased kernel function.
# If you give it a direction, it presumes you want it fixed. if you only provide the wave number, it presumes the direction is also variable.

function DirectionalKernel{T}(k::U, v::AbstractVector{<:Real}) where {T<:AbstractFloat, U<:Real}
    return FixedDirectionSFKernel{T}(T(k), 20*rand(T, 1) .+ 10*one(T), T.(v/norm(v)))
end

function DirectionalKernel(k::T, v::AbstractVector{<:Real}) where {T<:AbstractFloat}
    return FixedDirectionSFKernel{T}(T(k), 20*rand(T, 1) .+ 10*one(T), T.(v/norm(v)))
end

function DirectionalKernel{T}(k::U) where {T<:AbstractFloat, U<:Number}
    return VariableDirectionSFKernel{T}(T(k), 10*randn(T, 3) .+ one(T))
end

function DirectionalKernel(k::T) where {T<:AbstractFloat}
    return VariableDirectionSFKernel{T}(T(k), 10*randn(T, 3) .+ one(T))
end