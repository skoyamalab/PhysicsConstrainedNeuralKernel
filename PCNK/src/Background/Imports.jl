import Base: Float16, Float32, Float64, convert, eltype, show

# Functions meant to be overloaded for ease of use for converting the chains.

function Float16(a::Chain)
    return a|>f16
end

function Float32(a::Chain)
    return a|>f32
end

function Float64(a::Chain)
    return a|>f64
end

# How to get the eltype of the kernels:

function eltype(a::ISFKernel)
    return eltype(a.k)
end

Chain(D::NamedTuple) = Chain(D ...)