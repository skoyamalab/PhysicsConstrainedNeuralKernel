# Difference functions accounting for the various forms of inputs the kernels admit.

function __Diff(x1::AbstractVector{<:Number}, x2::AbstractVector{<:Number})
    return x1 - x2 
end

function __Diff(x1::AbstractMatrix{<:Number}, x2::AbstractVector{<:Number})
    return x1 .- x2
end

function __Diff(x1::AbstractVector{<:Number}, x2::AbstractMatrix{<:Number})
    return x1 .- x2
end

function __Diff(x1::AbstractMatrix{<:Number}, x2::AbstractMatrix{<:Number})
    B1 = size(x1, 2)
    B2 = size(x2, 2)
    return reshape(x1, :, B1, 1) .- reshape(x2, :, 1, B2)
end