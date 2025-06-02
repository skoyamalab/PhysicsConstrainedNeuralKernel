# This function does not have access to the implementations of the evaluation function, so only the dispatch remains.

function (a::CompositeKernel)(x::AbstractArray{<:Number})
    return a.AnalyticalKernel(x) + a.NeuralKernel(x)
end

function (a::CompositeKernel)(x1::AbstractVecOrMat{<:Number}, x2::AbstractVecOrMat{<:Number})
    return a.AnalyticalKernel(x1, x2) + a.NeuralKernel(x1, x2)
end