# Evaluation of the uniform kernel

function _ISFKernel_eval(a::UniformKernel, x::AbstractVector{<:Number})
    η = sqrt(sum(x.^2))
    return j0(a.k*η)
end

function _ISFKernel_eval(a::UniformKernel, x::AbstractMatrix{<:Number})
    η = sqrt.(dropdims(sum(x.^2, dims=1), dims=1))
    return j0.(a.k*η)
end

function _ISFKernel_eval(a::UniformKernel, x::AbstractArray{<:Number, 3})
    η = sqrt.(dropdims(sum(x.^2, dims=1), dims=1))
    return j0.(a.k*η)
end

# Dispatch with the various forms of inputs

function (a::fixed_UniformKernel)(x::AbstractArray{<:Number})
    return @inline _ISFKernel_eval(a, x)
end

function (a::scaled_UniformKernel)(x::AbstractArray{<:Number})
    return view(a.σ, 1) .* (@inline _ISFKernel_eval(a, x))
end

# Dispatch with two inputs as the kernel is meant to be used.

function (a::fixed_UniformKernel)(x1::AbstractVecOrMat{<:Number}, x2::AbstractVecOrMat{<:Number})
    Δx = @inline __Diff(x1, x2)
    return @inline _ISFKernel_eval(a, Δx)
end

function (a::scaled_UniformKernel)(x1::AbstractVecOrMat{<:Number}, x2::AbstractVecOrMat{<:Number})
    Δx = @inline __Diff(x1, x2)
    return view(a.σ, 1) .* (@inline _ISFKernel_eval(a, Δx))
end