# ========================= Multidirectional weight kernel formulations ================================
# =========== Kernels based on multiple kernel learning for various directional kernels ================
# These are infinite dimensional kernels that consider several bias directions simultaneously. The weight function is the result from a Von-Mises distribution.

abstract type MultiDirectionalKernel{T<:AbstractFloat, N<:Integer} <:DirectionalKernel{T} end

# Multidirectional weight kernel with fixed directions.

mutable struct FixedMultiDirectionSFKernel{T<:AbstractFloat, N<:Integer} <: MultiDirectionalKernel{T, N}
    k::T
    Ord::N
    β::AbstractVector{<:Number}
    γ::AbstractVector{<:Number}
    v::AbstractMatrix{T}
end

# Constructor

function FixedMultiDirectionSFKernel{T, N}(k::U, Ord::N2)::FixedMultiDirectionSFKernel{T, N} where{T<:AbstractFloat, U<:AbstractFloat, N<:Integer, N2<:Integer}
    x, y, z, w = lebedev_by_order(Ord)
    v = Matrix{T}([x y z]')
    β = 0.5rand(T, length(w)) .+ 10*one(T)
    return FixedMultiDirectionSFKernel{T, N}(T(k), N(Ord), β, T.(w), v)
end

function FixedMultiDirectionSFKernel{N}(k::T, Ord::N2)::FixedMultiDirectionSFKernel{T, N} where{T<:AbstractFloat, N<:Integer, N2<:Integer}
   x, y, z, w = lebedev_by_order(Ord)
   v = Matrix{T}([x y z]')
   β = 0.5rand(T, length(w)) .+ 10*one(T)
   return FixedMultiDirectionSFKernel{T, N}(T(k), N(Ord), β, T.(w), v)
end

function FixedMultiDirectionSFKernel{T}(k::U, Ord::N)::FixedMultiDirectionSFKernel{T, N} where{T<:AbstractFloat, U<:AbstractFloat, N<:Integer}
   x, y, z, w = lebedev_by_order(Ord)
   v = Matrix{T}([x y z]')
   β = 0.5rand(T, length(w)) .+ 10*one(T)
   return FixedMultiDirectionSFKernel{T, N}(T(k), Ord , β, T.(w), v)
end

function FixedMultiDirectionSFKernel(k::T, Ord::N)::FixedMultiDirectionSFKernel{T, N} where{T<:AbstractFloat, N<:Integer}
   x, y, z, w = lebedev_by_order(Ord)
   v = Matrix{T}([x y z]')
   β = 0.5rand(T, length(w)) .+ 10*one(T)
   return FixedMultiDirectionSFKernel{T, N}(k, Ord, β, T.(w), v)
end

# Implementations

function (a::FixedMultiDirectionSFKernel)(x::AbstractVector{<:Number})
   return sum( (a.γ ./ i0.(a.β)) .* j0.(sqrt.(reshape(sum((a.k*x .- im*(a.v .* a.β')).^2, dims=1), :))))
end

function (a::FixedMultiDirectionSFKernel)(X::AbstractMatrix{<:Number})::AbstractVector
   D = length(a.γ)
   d, B = size(X)
   return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k * X, (d, 1, B)), 1, D, 1) - im*repeat(reshape(a.v .* a.β', (d, D, 1)), 1, 1, B)).^2, dims=1), D, B))) .* (a.γ ./ i0.(a.β)), dims=1), :)
end

function (a::FixedMultiDirectionSFKernel)(ΔX::AbstractArray{<:Number, 3})::AbstractMatrix
   D = length(a.γ)
   d, B1, B2 = size(ΔX)
   return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k* ΔX, (d, 1, B1, B2)), 1, D, 1, 1) -im * repeat(reshape(a.v .* a.β', (d, D, 1, 1)), 1, 1, B1, B2)).^2, dims=1), D, B1, B2))) .* repeat(reshape(a.γ ./ i0.(a.β), (D, 1, 1)), 1, B1, B2), dims=1), (B1, B2))
end

function (a::FixedMultiDirectionSFKernel)(x1::AbstractVector{<:Number}, x2::AbstractVector{<:Number})
   return sum( (a.γ ./ i0.(a.β)) .* j0.(sqrt.(reshape(sum((a.k*(x1 - x2) .- im*(a.v .* a.β')).^2, dims=1), :))) )
end

function (a::FixedMultiDirectionSFKernel)(X1::AbstractMatrix{<:Number}, x2::AbstractVector{<:Number})::AbstractVector
   D = length(a.γ)
   d, B = size(X1)
   return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k * (X1 .- x2), (d, 1, B)), 1, D, 1) - im*repeat(reshape(a.v .* a.β', (d, D, 1)), 1, 1, B)).^2, dims=1), D, B))) .*(a.γ ./ i0.(a.β)), dims=1), :)
end

function (a::FixedMultiDirectionSFKernel)(x1::AbstractVector{<:Number}, X2::AbstractMatrix{<:Number})::AbstractVector
   D = length(a.γ)
   d, B = size(X2)
   return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k * (x1 .- X2), (d, 1, B)), 1, D, 1) - im*repeat(reshape(a.v .* a.β', (d, D, 1)), 1, 1, B)).^2, dims=1), D, B))) .*(a.γ ./ i0.(a.β)), dims=1), :)
end

function (a::FixedMultiDirectionSFKernel)(X1::AbstractMatrix{<:Number}, X2::AbstractMatrix{<:Number})::AbstractMatrix
   D = length(a.γ)
   @tullio ΔX[t, b1, b2] := X1[t, b1] - X2[t, b2]
   d, B1, B2 = size(ΔX)
   return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k* ΔX, (d, 1, B1, B2)), 1, D, 1, 1) -im * repeat(reshape(a.v .* a.β', (d, D, 1, 1)), 1, 1, B1, B2)).^2, dims=1), D, B1, B2))) .* repeat(reshape(a.γ ./ i0.(a.β), (D, 1, 1)), 1, B1, B2), dims=1), (B1, B2))
end

# Flux management

@functor FixedMultiDirectionSFKernel
trainable(a::FixedMultiDirectionSFKernel) = (;a.β, a.γ)

# Multidirectional weight kernel with fixed directions and fixed β values.
# Due to the normalization, the gradients of β can be very small, which makes learning unstable.
# For that reason, the value of β can be fixed by the user to stabilize training.

mutable struct FixedβMultiDirectionSFKernel{T<:AbstractFloat, N<:Integer} <: MultiDirectionalKernel{T, N}
   k::T
   Ord::N
   β::AbstractVector{T}
   γ::AbstractVector{<:Number}
   v::AbstractMatrix{T}
end

# Constructor

function FixedβMultiDirectionSFKernel{T, N}(k::Real, β::Real, Ord::Integer) where {T<:AbstractFloat, N<:Integer}
   x,y,z,w = lebedev_by_order(Ord)
   v = Matrix{T}([x y z]')
   γ = T.(w)
   βs = fill(T(β), length(γ))
   return FixedβMultiDirectionSFKernel{T, N}(T(k), N(Ord), βs, γ, v)
end

function FixedβMultiDirectionSFKernel{T, N}(k::Real, β::AbstractVector{<:Real}, Ord::Integer) where {T<:AbstractFloat, N<:Integer}
   x,y,z,w = lebedev_by_order(Ord)
   v = Matrix{T}([x y z]')
   γ = T.(w)
   βs = T.(β)
   return FixedβMultiDirectionSFKernel{T, N}(T(k), N(Ord), βs, γ, v)
end

function FixedβMultiDirectionSFKernel{N}(k::T, β::Real, Ord::Integer) where {T<:AbstractFloat, N<:Integer}
   x,y,z,w = lebedev_by_order(Ord)
   v = Matrix{T}([x y z]')
   γ = T.(w)
   βs = fill(T(β), length(γ))
   return FixedβMultiDirectionSFKernel{T, N}(T(k), N(Ord), βs, γ, v)
end

function FixedβMultiDirectionSFKernel{N}(k::T, β::AbstractVector{<:Real}, Ord::Integer) where {T<:AbstractFloat, N<:Integer}
   x,y,z,w = lebedev_by_order(Ord)
   v = Matrix{T}([x y z]')
   γ = T.(w)
   βs = T.(β)
   return FixedβMultiDirectionSFKernel{T, N}(T(k), N(Ord), βs, γ, v)
end

function FixedβMultiDirectionSFKernel(k::T, β::Real, Ord::N) where {T<:AbstractFloat, N<:Integer}
   x,y,z,w = lebedev_by_order(Ord)
   v = Matrix{T}([x y z]')
   γ = T.(w)
   βs = fill(T(β), length(γ))
   return FixedβMultiDirectionSFKernel{T, N}(T(k), N(Ord), βs, γ, v)
end

function FixedβMultiDirectionSFKernel(k::T, β::AbstractVector{<:Real}, Ord::N) where {T<:AbstractFloat, N<:Integer}
   x,y,z,w = lebedev_by_order(Ord)
   v = Matrix{T}([x y z]')
   γ = T.(w)
   βs = T.(β)
   return FixedβMultiDirectionSFKernel{T, N}(T(k), N(Ord), βs, γ, v)
end

# Implementations

function (a::FixedβMultiDirectionSFKernel)(x::AbstractVector{<:Number})
  return sum( (a.γ ./ i0.(a.β)) .* j0.(sqrt.(reshape(sum((a.k*x .- im*(a.v .* a.β')).^2, dims=1), :))))
end

function (a::FixedβMultiDirectionSFKernel)(X::AbstractMatrix{<:Number})
  D = length(a.γ)
  d, B = size(X)
  return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k * X, (d, 1, B)), 1, D, 1) - im*repeat(reshape(a.v .* a.β', (d, D, 1)), 1, 1, B)).^2, dims=1), D, B))) .* (a.γ ./ i0.(a.β)), dims=1), :)
end

function (a::FixedβMultiDirectionSFKernel)(ΔX::AbstractArray{<:Number, 3})
  D = length(a.γ)
  d, B1, B2 = size(ΔX)
  return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k* ΔX, (d, 1, B1, B2)), 1, D, 1, 1) -im * repeat(reshape(a.v .* a.β', (d, D, 1, 1)), 1, 1, B1, B2)).^2, dims=1), D, B1, B2))) .* repeat(reshape(a.γ ./ i0.(a.β), (D, 1, 1)), 1, B1, B2), dims=1), (B1, B2))
end

function (a::FixedβMultiDirectionSFKernel)(x1::AbstractVector{<:Number}, x2::AbstractVector{<:Number})
  return sum( (a.γ ./ i0.(a.β)) .* j0.(sqrt.(reshape(sum((a.k*(x1 - x2) .- im*(a.v .* a.β')).^2, dims=1), :))) )
end

function (a::FixedβMultiDirectionSFKernel)(X1::AbstractMatrix{<:Number}, x2::AbstractVector{<:Number})
  D = length(a.γ)
  d, B = size(X1)
  return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k * (X1 .- x2), (d, 1, B)), 1, D, 1) - im*repeat(reshape(a.v .* a.β', (d, D, 1)), 1, 1, B)).^2, dims=1), D, B))) .*(a.γ ./ i0.(a.β)), dims=1), :)
end

function (a::FixedβMultiDirectionSFKernel)(x1::AbstractVector{<:Number}, X2::AbstractMatrix{<:Number})
  D = length(a.γ)
  d, B = size(X2)
  return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k * (x1 .- X2), (d, 1, B)), 1, D, 1) - im*repeat(reshape(a.v .* a.β', (d, D, 1)), 1, 1, B)).^2, dims=1), D, B))) .*(a.γ ./ i0.(a.β)), dims=1), :)
end

function (a::FixedβMultiDirectionSFKernel)(X1::AbstractMatrix{<:Number}, X2::AbstractMatrix{<:Number})
  D = length(a.γ)
  @tullio ΔX[t, b1, b2] := X1[t, b1] - X2[t, b2]
  d, B1, B2 = size(ΔX)
  γβ = reshape(a.γ ./ i0.(a.β), (D, 1, 1))
  kΔxβ = sum(abs2, a.k*ΔX, dims=1) .- reshape(sum(abs2, a.v .* a.β', dims=1), (D, 1, 1))
  iβ = reshape(sum(identity, reshape(a.k* ΔX, (d, 1, B1, B2)) .* reshape(a.v .* a.β', (d, D, 1, 1)), dims=1), (D, B1, B2))
  jkΔxβ = j0.(sqrt.(kΔxβ .- iβ*im))
  return reshape(sum(jkΔxβ .* γβ, dims=1), (B1, B2))
end

# Flux management

@functor FixedβMultiDirectionSFKernel
trainable(a::FixedβMultiDirectionSFKernel) = (;a.γ)

# Multidirectional weight kernel with variable directions.
# Not recommended due to the possibility of the directions generating degenerative systems that might not even
# be proper kernel functions and the increased possibility of hitting saddle points and local minima.
# Implemented because, in ideal conditions, the directions can provide better estimations given constraints are applied.

mutable struct VariableMultiDirectionSFKernel{T<:AbstractFloat, N<:Integer} <: MultiDirectionalKernel{T, N}
    k::T
    Ord::N
    β::AbstractMatrix{<:Number}
    γ::AbstractVector{<:Number}
end

# Constructor

function VariableMultiDirectionSFKernel{T, N}(k::U, Ord::N2)::VariableMultiDirectionSFKernel{T, N} where{T<:AbstractFloat, U<:AbstractFloat, N<:Integer, N2<:Integer}
    x, y, z, w = lebedev_by_order(Ord)
    v = Matrix{T}([x y z]')
    β = v .* (0.5rand(T, 1, length(w)) .+ 10*one(T))
    return VariableMultiDirectionSFKernel{T, N}(T(k), N(Ord), β, T.(w))
end

function VariableMultiDirectionSFKernel{N}(k::T, Ord::N2)::VariableMultiDirectionSFKernel{T, N} where{T<:AbstractFloat, N<:Integer, N2<:Integer}
   x, y, z, w = lebedev_by_order(Ord)
   v = Matrix{T}([x y z]')
   β = v .* (0.5rand(T,1, length(w)) .+ 10*one(T))
   return VariableMultiDirectionSFKernel{T, N}(T(k), N(Ord), β, T.(w))
end

function VariableMultiDirectionSFKernel{T}(k::U, Ord::N)::VariableMultiDirectionSFKernel{T, N} where{T<:AbstractFloat, U<:AbstractFloat, N<:Integer}
   x, y, z, w = lebedev_by_order(Ord)
   v = Matrix{T}([x y z]')
   β = v .* (0.5rand(T, 1, length(w)) .+ 10*one(T))
   return VariableMultiDirectionSFKernel{T, N}(T(k), Ord , β, T.(w))
end

function VariableMultiDirectionSFKernel(k::T, Ord::N)::VariableMultiDirectionSFKernel{T, N} where{T<:AbstractFloat, N<:Integer}
   x, y, z, w = lebedev_by_order(Ord)
   v = Matrix{T}([x y z]')
   β = v .* (0.5rand(T, 1, length(w)) .+ 10*one(T))
   return VariableMultiDirectionSFKernel{T, N}(k, Ord, β, T.(w))
end

# Implementations

function (a::VariableMultiDirectionSFKernel)(x::AbstractVector{<:Number})
   β = sqrt.(reshape(sum(abs2, a.β, dims=1), :))
   return sum( (a.γ ./ i0.(β)) .* j0.(sqrt.(reshape(sum((a.k*x .- im*(a.β)).^2, dims=1), :))))
end

function (a::VariableMultiDirectionSFKernel)(X::AbstractMatrix{<:Number})::AbstractVector
   D = length(a.γ)
   d, B = size(X)
   β = sqrt.(reshape(sum(abs2, a.β, dims=1), :))
   return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k * X, (d, 1, B)), 1, D, 1) - im*repeat(reshape(a.β, (d, D, 1)), 1, 1, B)).^2, dims=1), D, B))) .* (a.γ ./ i0.(β)), dims=1), :)
end

function (a::VariableMultiDirectionSFKernel)(ΔX::AbstractArray{<:Number, 3})::AbstractMatrix
   D = length(a.γ)
   d, B1, B2 = size(ΔX)
   β = sqrt.(reshape(sum(abs2, a.β, dims=1), :))
   return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k* ΔX, (d, 1, B1, B2)), 1, D, 1, 1) -im * repeat(reshape(a.β, (d, D, 1, 1)), 1, 1, B1, B2)).^2, dims=1), D, B1, B2))) .* repeat(reshape(a.γ ./ i0.(β), (D, 1, 1)), 1, B1, B2), dims=1), (B1, B2))
end

function (a::VariableMultiDirectionSFKernel)(x1::AbstractVector{<:Number}, x2::AbstractVector{<:Number})
   β = sqrt.(reshape(sum(abs2, a.β, dims=1), :))
   return sum( (a.γ ./ i0.(β)) .* j0.(sqrt.(reshape(sum((a.k*(x1 - x2) .- im*(a.β)).^2, dims=1), :))) )
end

function (a::VariableMultiDirectionSFKernel)(X1::AbstractMatrix{<:Number}, x2::AbstractVector{<:Number})::AbstractVector
   D = length(a.γ)
   d, B = size(X1)
   β = sqrt.(reshape(sum(abs2, a.β, dims=1), :))
   return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k * (X1 .- x2), (d, 1, B)), 1, D, 1) - im*repeat(reshape(a.β, (d, D, 1)), 1, 1, B)).^2, dims=1), D, B))) .*(a.γ ./ i0.(β)), dims=1), :)
end

function (a::VariableMultiDirectionSFKernel)(x1::AbstractVector{<:Number}, X2::AbstractMatrix{<:Number})::AbstractVector
   D = length(a.γ)
   d, B = size(X2)
   β = sqrt.(reshape(sum(abs2, a.β, dims=1), :))
   return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k * (x1 .- X2), (d, 1, B)), 1, D, 1) - im*repeat(reshape(a.β, (d, D, 1)), 1, 1, B)).^2, dims=1), D, B))) .*(a.γ ./ i0.(β)), dims=1), :)
end

function (a::VariableMultiDirectionSFKernel)(X1::AbstractMatrix{<:Number}, X2::AbstractMatrix{<:Number})::AbstractMatrix
   D = length(a.γ)
   @tullio ΔX[t, b1, b2] := X1[t, b1] - X2[t, b2]
   d, B1, B2 = size(ΔX)
   β = sqrt.(reshape(sum(abs2, a.β, dims=1), :))
   return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k* ΔX, (d, 1, B1, B2)), 1, D, 1, 1) -im * repeat(reshape(a.β, (d, D, 1, 1)), 1, 1, B1, B2)).^2, dims=1), D, B1, B2))) .* repeat(reshape(a.γ ./ i0.(β), (D, 1, 1)), 1, B1, B2), dims=1), (B1, B2))
end

# Converters
# There should also be some way that allows the multidirectional kernels to be mutually converted.

# function FixedMultiDirectionSFKernel{T,N}(a::VariableMultiDirectionSFKernel)::FixedMultiDirectionSFKernel{T, N} where {T<:AbstractFloat, N<:Integer}
#    β = T.(reshape(sum(abs2, a.β, dims=1), :))
#    v = T.(typeof(a.β)(a.β ./ β'))
#    return FixedMultiDirectionSFKernel{T, N}(T.(a.k), N(a.Ord), β, T.(a.γ), v)
# end

# function FixedMultiDirectionSFKernel{T}(a::VariableMultiDirectionSFKernel{U, N})::FixedMultiDirectionSFKernel{T, N} where {T<:Number,U<:AbstractFloat, N<:Integer}
#    if T<: Integer
#       β = reshape(sum(abs2, a.β, dims=1), :)
#       v = typeof(a.β)(a.β ./ β')
#       return FixedMultiDirectionSFKernel{U, T}(a.k, T(a.Ord), β, a.γ, v)
#    elseif T<:AbstractFloat
#       β = T.(reshape(sum(abs2, a.β, dims=1), :))
#       v = T.(typeof(a.β)(a.β ./ β'))
#       return FixedMultiDirectionSFKernel{T, N}(T.(a.k), a.Ord, β, T.(a.γ), v)
#    else
#       println("These kernels require Float and Integer parameters. Given parameter is ", T)
#    end
# end


# function FixedMultiDirectionSFKernel(a::VariableMultiDirectionSFKernel{T, N})::FixedMultiDirectionSFKernel{T, N} where {T<:AbstractFloat, N<:Integer}
#    β = reshape(sum(abs2, a.β, dims=1), :)
#    v = typeof(a.β)(a.β ./ β')
#    return FixedMultiDirectionSFKernel{T, N}(a.k, a.Ord, β, a.γ, v)
# end

# function VariableMultiDirectionSFKernel{T,N}(a::FixedMultiDirectionSFKernel)::VariableMultiDirectionSFKernel{T, N} where {T<:AbstractFloat, N<:Integer}
#    β = T.(a.v .* a.β')
#    return VariableMultiDirectionSFKernel{T, N}(T.(a.k), N(a.Ord), β, T.(a.γ))
# end

# function VariableMultiDirectionSFKernel{T}(a::FixedMultiDirectionSFKernel{U, N})::VariableMultiDirectionSFKernel where {T<:Number,U<:AbstractFloat, N<:Integer}
#    if T<:Integer
#       β = a.v .* a.β'
#       return VariableMultiDirectionSFKernel{U, T}(a.k, T(a.Ord), β, a.γ)
#    elseif T<:AbstractFloat
#       β = T.(a.v .* a.β')
#       return VariableMultiDirectionSFKernel{T, N}(T.(a.k), a.Ord, β, T.(a.γ))
#    else
#       println("These kernels require Float and Integer parameters. Given parameter is ", T)
#    end
# end

# function VariableMultiDirectionSFKernel(a::FixedMultiDirectionSFKernel{T, N})::VariableMultiDirectionSFKernel{T, N} where {T<:AbstractFloat, N<:Integer}
#    β = a.v .* a.β'
#     return VariableMultiDirectionSFKernel{T, N}(a.k, a.Ord, β, a.γ)
# end

# Flux management

@functor VariableMultiDirectionSFKernel
trainable(a::VariableMultiDirectionSFKernel) = (;a.β, a.γ)

# Constructor for the multidirectional kernels defaults to the fixed direction variety.

function MultiDirectionalKernel{T}(k::U,Ord::N) where {T<:AbstractFloat, U<:Real, N<:Integer}
   return FixedMultiDirectionSFKernel{T}(k, Ord)
end

function MultiDirectionalKernel(k::T,Ord::N) where {T<:AbstractFloat, N<:Integer}
   return FixedMultiDirectionSFKernel{T}(k, Ord)
end

function MultiDirectionalKernel{T}(k::U,β::Real, Ord::N) where {T<:AbstractFloat, U<:Real, N<:Integer}
   return FixedβMultiDirectionSFKernel{T, N}(k, β, Ord)
end

function MultiDirectionalKernel(k::T,β::Real, Ord::N) where {T<:AbstractFloat, N<:Integer}
   return FixedβMultiDirectionSFKernel{T, N}(k, β, Ord)
end

function MultiDirectionalKernel{T}(k::U,β::AbstractVector{<:Real}, Ord::N) where {T<:AbstractFloat, U<:Real, N<:Integer}
   return FixedβMultiDirectionSFKernel{T, N}(k, β, Ord)
end

function MultiDirectionalKernel(k::T,β::AbstractVector{<:Real}, Ord::N) where {T<:AbstractFloat, N<:Integer}
   return FixedβMultiDirectionSFKernel{T, N}(k, β, Ord)
end

