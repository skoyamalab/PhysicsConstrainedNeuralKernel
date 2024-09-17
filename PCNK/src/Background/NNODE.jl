# Since DiffEqFlux no longer supports Flux, only Lux, I made a version that will work well enough for this use case.
# Future versions should aim to support Lux, although I am not sure how explicit parameters will work with the kernel functions.

# The structure of a NODE layer, which requires basic info for the solving of the ODE.

struct NODE{T<:AbstractFloat}
    pars::AbstractVector
    reconstructor::Restructure
    timespan::Tuple{T,T}
    solver
end

#Constructors of the NODE layer, which use the proposed architecture of the NN derivative to automatically derive the ouput

function NODE{T}(dW, timespan::Tuple{<:Real, <:Real}=(T(0), T(1)), solver = Tsit5()) where {T<:AbstractFloat}
    p, re = destructure(dW)# It is necessary to break down the NN in order to extract the gradients explicitly.
    return NODE{T}(p, re, (T.(timespan[1]), T.(timespan[2])), solver)
end

function NODE(dW, timespan::Tuple{<:Real, <:Real}=(Float32(0), Float32(1)), solver = Tsit5())
    return NODE{Float32}(dW, (Float32(timespan[1]), Float32(timespan[2])), solver)
end

function (a::NODE)(x::AbstractArray)
    dxdt(x, p, t) = a.reconstructor(p)(x)
    prob = ODEProblem(dxdt, x, a.timespan, a.pars)
    return last(solve(prob, a.solver, saveat = a.timespan[2]))
end

#The only real trainable parameters of the NODE layer are the parameters needed to reconstruct the NN
@functor NODE
trainable(a::NODE) = (; a.pars)
