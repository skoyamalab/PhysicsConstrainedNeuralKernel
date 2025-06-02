function convert(T::DataType, a::PlaneWavePINKernel)
    return PlaneWavePINKernel{T}(convert(T, a.AnalyticalKernel), convert(T, a.NeuralKernel))
end

function convert(T::DataType, a::DirectedResidualPINKernel)
    return DirectedResidualPINKernel{T}(convert(T, a.AnalyticalKernel), convert(T, a.NeuralKernel))
end
