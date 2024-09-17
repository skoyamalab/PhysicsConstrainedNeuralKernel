# ========================= Resampler for the various neural-based kernels that support it =============================

# Perhaps the most impotant differentiating factor between the neural-based and the parameter-based kernels:
# the neural kernel has the ability to make its representation arbitrarily more complex 

function resample(a::NeuralISFKernel{T, N}, Ord::N2) where {T<:AbstractFloat, N<:Integer, N2<:Integer}
    @ignore_derivatives a.Ord = N(Ord)
    x, y, z, w = lebedev_by_order(Ord)
    @ignore_derivatives a.w = typeof(a.w)(w)
    @ignore_derivatives a.v = typeof(a.v)([x y z]')
    return nothing
end


function resample(a::CompositeKernel, Ord::N)::nothing where {N<:Integer}
    resample(a.NeuralKernel, Ord)
    return nothing
end