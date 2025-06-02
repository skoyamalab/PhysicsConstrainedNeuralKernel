# Implementation function for the single direction case

function _ISFKernel_eval(a::DirectionalSFKernel{<:AbstractFloat, <:AbstractVector}, x::AbstractVector{<:Number})
    β = view(a.β, 1)
    return view(a.σ, 1) .* j0.(sqrt(sum((Complex.(a.k*x) - im*(β .*a.v)).^2)))/i0.(β)
end

function _ISFKernel_eval(a::DirectionalSFKernel{<:AbstractFloat, <:AbstractVector}, x::AbstractMatrix{<:Number})
    β = view(a.β, 1)
    return view(a.σ, 1) .* j0.(sqrt.(reshape(sum((Complex.(a.k*x) .- im*(β .*a.v)).^2, dims=1), :)))/i0.(β)
end

function _ISFKernel_eval(a::DirectionalSFKernel{<:AbstractFloat, <:AbstractVector}, x::AbstractArray{<:Number, 3})
    d, B1, B2 = size(x)
    β = view(a.β, 1)
    return view(a.σ, 1) .* j0.(sqrt.(reshape(sum((a.k*x  - im*repeat(reshape(β .*a.v, (d, 1, 1)), 1, B1, B2)).^2, dims=1), B1, B2)))/i0.(β)
end

# Implementations for the multiple direction case
function _ISFKernel_eval(a::DirectionalSFKernel{<:AbstractFloat, <:AbstractMatrix}, x::AbstractVector{<:Number})
   return sum( (a.σ ./ i0.(a.β)) .* j0.(sqrt.(reshape(sum((a.k*x .- im*(a.v .* reshape(a.β, 1, :))).^2, dims=1), :))))
end

function _ISFKernel_eval(a::DirectionalSFKernel{<:AbstractFloat, <:AbstractMatrix}, x::AbstractMatrix{<:Number})
   D = length(a.σ)
   d, B = size(x)
   return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k * x, (d, 1, B)), 1, D, 1) - im*repeat(reshape(a.v .* a.β', (d, D, 1)), 1, 1, B)).^2, dims=1), D, B))) .* (a.σ ./ i0.(a.β)), dims=1), :)
end

function _ISFKernel_eval(a::DirectionalSFKernel{<:AbstractFloat, <:AbstractMatrix}, x::AbstractArray{<:Number, 3})
   D = length(a.σ)
   d, B1, B2 = size(x)
   return reshape(sum(j0.(sqrt.(reshape(sum((repeat(reshape(a.k* x, (d, 1, B1, B2)), 1, D, 1, 1) -im * repeat(reshape(a.v .* a.β', (d, D, 1, 1)), 1, 1, B1, B2)).^2, dims=1), D, B1, B2))) .* repeat(reshape(a.σ ./ i0.(a.β), (D, 1, 1)), 1, B1, B2), dims=1), (B1, B2))
end

# Dispatches of the kernel objects using the implementation functions
function (a::DirectionalSFKernel)(x::AbstractArray{<:Real})
    return @inline _ISFKernel_eval(a, x)
end

function (a::DirectionalSFKernel)(x1::AbstractVecOrMat{<:Real}, x2::AbstractVecOrMat{<:Real})
    return @inline _ISFKernel_eval(a, __Diff(x1, x2))
end