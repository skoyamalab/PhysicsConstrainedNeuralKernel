# Benchmarking library included for further testing. Comment if unneeded.
using BenchmarkTools
# Libraries needed to compile the PCNK module. The overhead can be eliminated by performing static compilation.
using GenericLinearAlgebra, LinearAlgebra, Polynomials, SpecialFunctions, ArrayAllocators, SphericalHarmonics, KernelFunctions, ChainRules, JLD2,Lebedev, Functors, Optimization, Optimisers, FFTW, Flux, JLD2, KernelAbstractions, OrdinaryDiffEq
using CUDA
using Tullio
using OrdinaryDiffEq, SciMLSensitivity
# Find current directory.
DIR = @__DIR__
# The script pressuposes the folder PCNK with the library is in the same directory as the script.
include(string(DIR, "/PCNK/master.jl"))
# Include module
using .PCNK

# Precision. Generally stick to 32 bit when using Flux models, as that is what most operations default to.
T = Float32
T60 = "200ms"

# Training data
data = JLD2.load(string(DIR, "/", T60,"/TASLP.jld"))
mics_xyz = data["xyz_rec"];
freqs = data["freqs"];
c = T(343)
ks = T.(2*π*freqs/c)
Ord_dir = 5
Ord_NN = 7
# Default value for regularization constant, given as variance of the noise for a 20dB SNR.
λ = 0.01
# Load simulated recordings
ir = data["recordings"];

# This program was run several times with varying input values, the above input data represents the the best results over each frequency

# Validation data for frequency performance Evaluation_set
mics_val = data["xyz_val"]
ir_val = data["validation"]


#Initial definition of the kernels (wavenumber is placeholder that is replaced in each iteration)
k = ks[2]
κ_uni = UniformKernel{T}(k)
κ_prop = DirectedResidualKernel{T}(k, Ord_dir, Ord_NN)
_, κ = destructure(κ_prop)
w_max =0.15# Maximum value for residual kernel part
# Seeds of random initialization that generated best results
Χ00 = data["best seeds"]
#Uncomment in order to try a random seed
# χ00, κ = destructure(κ_prop)
# Χ00 = Matrix{T}(calloc, length(χ00), 11)
# Χ00[:,:] .= χ00 
# # It is necessary to correct the last layer of the residual field to ensure the initial value is an interior point. It depends on the architecture of the neural weight.
# if sum(κ_prop.NeuralKernel.W(ks[end] * κ_prop.NeuralKernel.v) .* κ_prop.NeuralKernel.w') > w_max
#     Χ00[end-4:end, :] *= atanh(w_max) - eps(T)
# end

#The expected contribution of the residual component is relatively small.
Nx = size(Χ00, 1)

# Training microphone positions on CPU as the models are not large enough to justify training on GPU.
# Testing microphone positions on GPU for faster deployment.
X = T.(mics_xyz)
X_val = T.(mics_val)
X_gpu = T.(mics_xyz)|>gpu
X_val_gpu = T.(mics_val)|>gpu


# Much like MATLAB, it is faster in Julia to prealocate values that will be stored.
coeffs_uni = Matrix{Complex{T}}(calloc, size(X, 2), length(freqs))
err_uni = zeros(T, length(freqs))

#Added only to display performance, hence being added late into the file.
using ProgressMeter

@showprogress for f=eachindex(ks)
    # Set wave number to desired value
    κ_uni.k = ks[f]
    # Set output to correspond with frequency
    y = ir[:, f]
    y_val = ir_val[:, f]
    # Calculate batch-to-batch of testing data with itself to have Gram matrix
    K = κ_uni(X,X)
    α = (K + λ*I)\y
    # Copy coefficient to CPU to store them
    coeffs_uni[:, f] = α|>cpu
    # Perform estimation
    y_est = (κ_uni)(X_val, X)*(α)
    # Store error
    err_uni[f] = sum(abs2, y_val-y_est)/sum(abs2, y_val)
end

# Define general training loss function, here made to take in a kernel constructor and kernel parameters.
# The function can be defined quite freely, as the gradients will be calculated with AD.
# Note that AD compatibility and performance does depend on the loss.
function ℓ(x::AbstractMatrix,y::AbstractArray,κ::Optimisers.Restructure, ps)
    kernel = κ(ps)
    K = kernel(x,x)
    α = (K + λ*I)\y
    return sum(real, conj(y) .* α)
end

# Optimization metrics to evaluate. The value the parameters take per iteration, the kernel expansion coefficients and the aggregate error per frequency.
Χ = Matrix{T}(calloc, Nx, length(freqs))
coeffs_prop = Matrix{Complex{T}}(calloc, size(X, 2), length(freqs))
err_prop = Vector{T}(calloc, length(freqs))

# We will optimize using a constraint-aware optimizer, which displays how the kernels can be trained using a variety of techniques.
# We will also use autodiff for the training, meaning there is no need to consider how the input changes the output.

## Constraint function, which says the constraints as they relate to the parameters.
# We limit the output of the residual part as to make sure no overfitting occurs
function cons!(constraint, χ, y)
    kernel = κ(χ)
    res_ampl = sum(kernel.NeuralKernel.W(kernel.NeuralKernel.k * kernel.NeuralKernel.v).*kernel.NeuralKernel.w')
    constraint .= [sum(χ[15:28]);res_ampl; χ[1:28]]
end
## The function is an inplace function considering optimization input (Χ) and parameters (y). 
## Lower bound of the constraints: the sum is exactly 1, so the lower bound should be
## Σγ≥1, γ .≥ 0, β .≥0 and there is no lower bound to observe on the residual part(the NN architecture guarantees it is nonnegative).
lcons = [1.0;-Inf; fill(zero(T), 28)]
## Upper bound of the constraints: the sum is exactly equal to 1, while the upper bound γ is not considered.
## We set it to ∞, however informing this to Julia simply means there is no constraint considered.
## Σγ≤1, γ .≤ ∞, β .≤ ∞ and the neural part has a maximum of 0.2.
ucons = [1.0;w_max;fill(T.(Inf), 28)]
# Any derivatives of the constraints are calculated by the optimizer itself, which is using autodiff, so this is sufficient to guarantee they are satisfied.
# Note that setting constraints to -∞ or ∞ only means the constraints will be ignored.

# We are optimizing the problem using the Optim library included in Optimization.jl, and will be using Zygote (generally the most supported AD framework in Julia).
using OptimizationOptimJL, ForwardDiff

# Training of the kernel itself
@showprogress for f=1:length(freqs)
    # First step: set the wavenumber of the kernel(s)
    κ_prop.AnalyticalKernel.k = ks[f]
    κ_prop.NeuralKernel.k = ks[f]
    # Since the Julia ecosystem tends to prefer explicit variables, it is recommended to use a Optimisers.Restructure object.
    # Implicit training using Flux.params is still possible.
    _, κ = destructure(κ_prop)
    # Set output to correspond with frequency
    y = ir[:, f]# Derivation data
    y_val = ir_val[:, f]# Validation data
    # Set initial values for the network parameters and normalize.
    χ = Float64.(Χ00[:, f])
    χ[15:28] /= sum(χ[15:28])
    # Define loss function for optimization
    loss(χ, y) = ℓ(X, y, κ, χ)
    # This function 
    # Define optimization function as well as what method of differentiation will be used and the constraints.
    func = Optimization.OptimizationFunction(loss,
    Optimization.AutoZygote(),
    cons = cons!)
    # Define the optimization problem, which includes the function that must be optimized and the lower and upper bounds of the constraint.
    prob = Optimization.OptimizationProblem(func, χ, y, lcons = lcons, ucons=ucons)
    # Derive a solution using an appropriate optimizer.
    sol = Optimization.solve(prob, IPNewton())
    # The solution object can be treated as a vector and we can now extract its value.
    χ = T.(sol)
    # Eliminate any numerical resquice placed by the optimizer
    χ[abs.(χ) .< eps(T)] .= zero(T)
    χ[15:28][χ[15:28] .< eps(T)] .=0
    # Now, we can reconstruct the kernel.
    κ_eval = κ(χ)
    K = κ_eval(X,X)
    # We can now calculate the kernel coefficients associated with each frequency
    α = (K+λ*I)\(y)
    #We now store the coefficients of the kernel regression
    coeffs_prop[:,f] = α
    Χ[:, f] = χ
    y_est = κ_eval(X_val, X)*α
    err_prop[f] = sum(abs2, y_est - y_val )/sum(abs2, y_val)
end

Χ

# We can compare the error of the coarse grid complete model and the uniform kernel.
10log10.(err_uni)
10log10.(err_prop)

# We copy the previous coarse grid results 
err_prop_0 = copy(err_prop)
coeffs_prop_0 = copy(coeffs_prop)
# Refine the grid and increase system complexity without changing the parameter count.
resample(κ_prop.NeuralKernel, 17)
# Then we recalculate the errors
for f = 1:length(freqs)
    k = ks[f]
    κ_prop.AnalyticalKernel.k = ks[f]
    κ_prop.NeuralKernel.k = ks[f]
    _, κ = destructure(κ_prop)
    κ_eval = κ(Χ[:, f])
    y = ir[:, f]
    y_val = ir_val[:, f]
    K = κ_eval(X,X)
    coeffs_prop[:, f] = (K + λ*I)\y
    y_est = κ_eval(X_val, X)*coeffs_prop[:, f]
    err_prop[f] = sum(abs2, y_val - y_est)/sum(abs2, y_val)
end

ERR = [10log10.(err_uni[2:end]) 10log10.(err_prop_0[2:end]) 10log10.(err_prop[2:end])]

using Plots
# Normalized mean square error between both kernels.
FS = 18# Font size
LW = 3# Line width
colors = palette(:default)
p = plot(freqs[2:end],# Horizontal axis data
         ERR,
         linewidth=LW,
         color = [colors[3] colors[2] colors[1]],
         label = ["Uniform" "PCNK (coarse grid)" "PCNK (fine grid)"],
         style = [:dash :dot :solid],
         legend = :bottomright,
         xlabel = "Frequency (Hz)",
         ylabel = "NMSE (dB)",
         xlabelfontsize = FS,
         ylabelfontsize = FS,
         legendfontsize = FS-4,
         xtickfontsize = FS,
         ytickfontsize = FS,
         xticks = 200:400:2000
         )
# Save to GRAPHS folder
savefig(p, string(DIR, "/", T60, "/GRAPHS/NMSE(TASLP).pdf"))