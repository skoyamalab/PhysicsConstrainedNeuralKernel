# Promotion functions that will convert and promote the uniform kernel to a specific precision or type. This is to make easier to conciliate with the composite kernel.

# Conversion of the fixed factor kernel into the trainable factor kernel:

function promote_to_scaled(a::fixed_UniformKernel{T}) where {T<:AbstractFloat}
    return scaled_UniformKernel{T}(a.k)
end

function promote_to_scaled(a::scaled_UniformKernel)
    return a
end

# Convert a uniform kernel to the specific level of precision.

function convert(T::DataType, a::fixed_UniformKernel)
    return fixed_UniformKernel{float(T)}(a.k)
end

function convert(T::DataType, a::scaled_UniformKernel)
    sigma = 1 .* view(a.σ, 1)
    return scaled_UniformKernel{float(T)}(a.k; σ = sigma)
end
