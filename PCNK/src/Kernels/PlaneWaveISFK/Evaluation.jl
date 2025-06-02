# Generic evaluation function for the plane wave kernels

function _ISFKernel_eval(a::PlaneWaveKernel, x::AbstractVector{<:Number}, v::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real})
    kx = dropdims(sum(a.k*(x .* v), dims=1), dims = 1)
    return sum(exp.(im*kx) .* σ)
end

function _ISFKernel_eval(a::PlaneWaveKernel, X::AbstractMatrix{<:Number}, v::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real})
    B = size(X, 2)
    D = size(v, 2)
    kx = reshape(sum(reshape(v, :, D, 1)  .* reshape(X, :, 1, B), dims=1), :, B)
    return dropdims(sum(σ .* exp.(im*a.k*kx), dims=1), dims = 1)
end

function _ISFKernel_eval(a::PlaneWaveKernel, X::AbstractArray{<:Number, 3}, v::AbstractMatrix{<:Real}, σ::AbstractVector{<:Real})
    _, B1, B2 = size(X)
    D = size(v, 2)
    kx = reshape(sum(reshape(v, :, D, 1, 1) .* reshape(X, :, 1, B1, B2), dims=1), D, B1, B2)
    return dropdims(sum(σ .* exp.(im*a.k*kx), dims=1), dims = 1)
end

# Dispatch of the kernels for one input

function (a::DiscreteWeightPlaneWaveKernel)(x::AbstractArray{<:Real})
    return @inline _ISFKernel_eval(a, x, a.v, a.σ)
end

function (a::NeuralWeightPlaneWaveKernel)(x::AbstractArray{<:Real})
    σ = a.σ .* dropdims(a.W(a.k * a.v), dims = 1)
    return @inline _ISFKernel_eval(a, x, a.v, σ)
end

# Dispatch of the kernels for two inputs

function (a::DiscreteWeightPlaneWaveKernel)(x1::AbstractVecOrMat{<:Real}, x2::AbstractVecOrMat{<:Real})
    Δx = __Diff(x1, x2)
    return @inline _ISFKernel_eval(a, Δx, a.v, a.σ)
end

function (a::NeuralWeightPlaneWaveKernel)(x1::AbstractVecOrMat{<:Real}, x2::AbstractVecOrMat{<:Real})
    Δx = __Diff(x1, x2)
    σ = a.σ .* dropdims(a.W(a.k * a.v), dims = 1)
    return @inline _ISFKernel_eval(a, Δx, a.v, σ)
end