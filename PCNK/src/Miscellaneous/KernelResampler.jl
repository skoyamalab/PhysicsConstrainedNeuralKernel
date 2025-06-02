# ========================= Resampler for the various neural-based kernels that support it =============================

# Perhaps the most impotant differentiating factor between the neural-based and the parameter-based kernels:
# the neural kernel has the ability to make its representation arbitrarily more complex 

function resample!(a::NeuralWeightPlaneWaveKernel{T}, Ord::N) where {T<:AbstractFloat, N<:Integer}
    x, y, z, w = lebedev_by_order(Ord)
    @ignore_derivatives a.σ = typeof(a.σ)(w)
    @ignore_derivatives a.v = typeof(a.v)([x y z]')
    return nothing
end


function resample!(a::CompositeKernel, Ord::N)::nothing where {N<:Integer}
    resample!(a.NeuralKernel, Ord)
    return nothing
end