# Convert a uniform kernel to the specific level of precision.

function convert(T::DataType, a::FixedDirectionPlaneWaveKernel)
    return FixedDirectionPlaneWaveKernel{float(T)}(a.k, a.v, σ = a.σ)
end

function convert(T::DataType, a::TrainableDirectionPlaneWaveKernel)
    return TrainableDirectionPlaneWaveKernel{float(T)}(a.k, a.v, σ = a.σ)
end

function convert(T::DataType, a::NeuralWeightPlaneWaveKernel)
    return NeuralWeightPlaneWaveKernel{float(T)}(a.k, a.v, σ = a.σ, W = a.W)
end