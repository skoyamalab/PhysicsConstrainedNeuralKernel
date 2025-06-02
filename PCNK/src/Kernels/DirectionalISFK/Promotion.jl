function convert(T::DataType, a::TrainableDirectionSFKernel)
    k = T(a.k)
    v = T.(a.v)
    β = T.(a.β)
    σ = T.(a.σ)
    return TrainableDirectionSFKernel(k, σ, β, v)
end

function convert(T::DataType, a::FixedDirectionSFKernel)
    k = T(a.k)
    v = T.(a.v)
    β = T.(a.β)
    σ = T.(a.σ)
    return FixedDirectionSFKernel(k, σ, β, v)
end