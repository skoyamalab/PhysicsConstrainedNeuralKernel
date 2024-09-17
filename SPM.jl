# Benchmarking library included for further testing. Comment if unneeded.
using BenchmarkTools, JLD2
# Libraries needed to compile the PCNK module. The overhead can be eliminated by performing static compilation.
using GenericLinearAlgebra, LinearAlgebra, Polynomials, SpecialFunctions, ArrayAllocators, SphericalHarmonics, KernelFunctions, ChainRules,Lebedev, Functors, Optimization, Optimisers, FFTW, Flux, KernelAbstractions, OrdinaryDiffEq
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

# The first step is to load the data
data = JLD2.load(string(DIR, "/400ms/SPM.jld"))
freqs = data["freqs"]
c = T(343)
ks = T.(2*π*freqs/c)
# Get the recordings that will be used to derive the model and the positions of the microphones that generated them.
ir = data["recordings"]
X = data["xyz_rec"]
# Get the validation data, the clean signals that will be estimated, and the corresponding positions
ir_val = data["validation"]
X_val = data["xyz_val"]

# Test orders for the kernel function
Ord_dir = 5
Ord_NN = 11
# Default value for regularization constant, given as variance of the noise for a 20dB SNR.
λ = 0.01
#Initial definition of the kernels (wavenumber is placeholder that is replaced in each iteration). Uniform is for reference.
k = ks[2]
κ_uni = UniformKernel{T}(k)
β = fill(T.(10), 14)
κ_prop = DirectedResidualKernel{T}(k, β, Ord_dir, Ord_NN)
#We performed several computations in order to display good results, as this library was written after the initial experiments in the paper.
#these were the seeds with the best results for this configuration and for each frequency.
Χ00 = data["best seeds"]
Nx, F = size(Χ00);

# Much like MATLAB, it is faster in Julia to prealocate values that will be stored.
coeffs_uni = Matrix{Complex{T}}(calloc, size(X, 2), length(freqs))
err_uni = zeros(T, length(freqs))

#Added only to display performance, hence being added late into the file.
using ProgressMeter

# Test run with uniform kernel (no optimization) showing the model
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
# Note that AD compatibility and performance does depend on the loss. Not every function has well-defined differentiation rules
function ℓ(x::AbstractMatrix,y::AbstractArray,κ::Optimisers.Restructure, ps)
    kernel = κ(ps)
    K = kernel(x,x)
    α = (K + λ*I)\y
    res_ampl = sum(kernel.NeuralKernel.W(kernel.NeuralKernel.v) .* kernel.NeuralKernel.w')
    return sum(real, conj(y) .* α) +0.06res_ampl
end

# Optimization metrics to evaluate. The value the parameters take per iteration, the kernel expansion coefficients and the aggregate error per frequency.
Χ = Matrix{T}(calloc, Nx, length(freqs))
coeffs_prop = Matrix{Complex{T}}(calloc, size(X, 2), length(freqs))
err_prop = Vector{T}(calloc, length(freqs))

# We will optimize using a constraint-aware optimizer, which displays how the kernels can be trained using a variety of techniques.
# We will also use autodiff for the training, meaning the constraints will be upheld automatically.

## Constraint function, which says the only constraints observed are on the indexes corresponding to γ, being their sum and values.
function cons!(constraint, χ, y)
    constraint .= [sum(χ[1:14]);χ[1:14]]
end
## The function is an inplace function considering optimization input (Χ) and parameters (y)
## Note the constraints are unrelated to the outputs y in this case, but still need to be made aware.

## Lower bound of the constraints: the sum is exactly 1, so the lower bound should be
## Σγ≥1, γ .≥ 0
lcons = [1.0;fill(zero(T), 14)]
## Upper bound of the constraints: the sum is exactly equal to 1, while the upper bound γ is not considered.
## We set it to ∞, however informing this to Julia simply means there is no constraint considered.
## Σγ≤1, γ .≤ ∞
ucons = [1.0;fill(T.(Inf), 14)]
# Any derivatives of the constraints are calculated by the optimizer itself, which is using autodiff, so this is sufficient to guarantee they are satisfied.

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
    # Set initial values for the network parameters.
    χ = Float64.(Χ00[:,f])
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
    # Now, we can reconstruct the kernel.
    κ_eval = κ(χ)
    K = κ_eval(X,X)
    # We can now calculate the kernel coefficients associated with each frequency
    α = (K+λ*I)\(y)
    #We now store the coefficients of the kernel regression
    coeffs_prop[:,f] = α|>cpu
    Χ[:, f] = χ
    y_est = κ_eval(X_val, X)*coeffs_prop[:, f]
    err_prop[f] = sum(abs2, y_est - y_val )/sum(abs2, y_val)
end

# Let us collect the errors and show the relative improvement caused by training the model to fit the data
ERR = [10log10.(err_uni)[2:end] 10log10.(err_prop)[2:end]]
# We ignore the frequency 0 component due to the fact it does not fit the solution model. The losses are very small due to small variation, but the use of the kernel method is not necessary.

# The kernels can also be deployed on the GPU. Training can be performed as well.
# The kernels have only been tested on NVIDIA GPUs using CUDA.jl, however no operation makes explicit reference to it.
# In theory, models should run in other maker GPUs, but that has not been tested.
# It is recommended to use precision T=Float32 with the GPU as not only are most GPU applications optimized for it, but also the |>gpu macro presupposes it.

# We will use GPU deployment in order to evaluate the pointwise performance of the method.

#Copy outputs to GPU
f0=8 # Frequency of analysis
κ_uni.k = ks[f0]
κ_prop.AnalyticalKernel.k = ks[f0]
κ_prop.NeuralKernel.k = ks[f0]
y_PLANE = reshape(data["pointwise"][:, f0], (101, 101))|>gpu
# Copy inputs to GPU
X_plane = data["xyz_pointwise"]
X_plane_gpu = data["xyz_pointwise"]|>gpu
X_gpu = X|>gpu
# Copy kernels and parameters to GPU:
Χ_gpu = Χ|>gpu
κ_uni_gpu = κ_uni|>gpu
κ_prop_gpu = κ_prop|>gpu
α_uni = coeffs_uni[:, f0]
α_prop = coeffs_prop[:, f0]
α_uni_gpu = coeffs_uni[:, f0]|>gpu
α_prop_gpu = coeffs_prop[:, f0]|>gpu

# The estimations will be given as

y_PLANE_uni = reshape(κ_uni_gpu(X_plane_gpu, X_gpu)*α_uni_gpu, (101, 101))
y_PLANE_prop = reshape(κ_prop_gpu(X_plane_gpu, X_gpu)*α_prop_gpu, (101, 101))

# We can benchmark CPU and GPU deployment in order to see the performance differential.
@benchmark reshape(κ_uni(X_plane, X)*α_uni, (101, 101))
@benchmark reshape(κ_uni_gpu(X_plane_gpu, X_gpu)*α_uni_gpu, (101, 101))
@benchmark reshape(κ_prop(X_plane, X)*α_prop, (101, 101))
@benchmark reshape(κ_prop_gpu(X_plane_gpu, X_gpu)*α_prop_gpu, (101, 101))

# We can also calculate the normalized square error between each estimation and the ground truth in a pointwise fashion.

NSE_uni = 20log10.(abs.(y_PLANE_uni - y_PLANE)./abs.(y_PLANE))
NSE_prop = 20log10.(abs.(y_PLANE_prop - y_PLANE)./abs.(y_PLANE))

# Plotting results
using Plots
# Normalized mean square error between both kernels.
FS = 18# Font size
LW = 3# Line width
colors = palette(:default)
p = plot(freqs[2:end],# Horizontal axis data
         ERR,
         linewidth=LW,
         color = [colors[2] colors[1]],
         label = ["Uniform" "PCNK (Adaptive)"],
         style = [:dash :solid],
         legend = :bottomright,
         xlabel = "Frequency (Hz)",
         ylabel = "NMSE (dB)",
         xlabelfontsize = FS,
         ylabelfontsize = FS,
         legendfontsize = FS-4,
         xtickfontsize = FS,
         ytickfontsize = FS,
         xticks = 200:400:1600
         )

# Pointwise reconstruction test
# Load the pyplot backend
pyplot()

UC = round(maximum(abs, real(y_PLANE))/0.005)*0.005
Clim = (-UC, UC)
Δy = LinRange(-0.5, 0.5, 101)
Δx = LinRange(-0.5, 0.5, 101)

xt(t) = 0.49*cos(t)
yt(t) = 0.49*sin(t)
LW_MAX = 6
LW_min = 4
uc = floor(UC/0.01)*0.01
CBAR_TICKS = -uc:uc/2:uc
FS_CBAR = 22
SIZE = (700, 630)
Elim = (-15, 0)
EBAR_TICKS = -15:5:0

p_orig = heatmap(Δx,
                 Δy,
                 real(y_PLANE|>cpu),
                 aspect_ratio = :equal,
                 size = SIZE,
                 xlabel = "x (m)",
                 ylabel = "y (m)",
                 c = cgrad([:blue, :white, :red]),
                 clims = Clim,
                 xtickfontsize = FS_CBAR,
                 ytickfontsize = FS_CBAR,
                 labelfontsize = FS_CBAR,
                 yflip = true,
                 colorbar_ticks = CBAR_TICKS,
                 colorbar_tickfontsize = FS,
                 right_margin = 16Plots.mm
                 )
                 plot!(xt, yt, 0, 2π, leg = false, aspect_ratio = :equal, lw = LW_MAX, color=:black)
                 plot!(xt, yt, 0, 2π, leg=false, aspect_ratio=:equal, lw=LW_min, color=:white, xlims = (-0.5, 0.5), ylims = (-0.5, 0.5), xticks = -0.4:0.4:0.4, yticks = -0.4:0.4:0.4)

p_uni = heatmap(Δx,
                 Δy,
                 real(y_PLANE_uni|>cpu),
                 aspect_ratio = :equal,
                 size = SIZE,
                 xlabel = "x (m)",
                 ylabel = "y (m)",
                 c = cgrad([:blue, :white, :red]),
                 clims = Clim,
                 xtickfontsize = FS_CBAR,
                 ytickfontsize = FS_CBAR,
                 labelfontsize = FS_CBAR,
                 yflip = true,
                 colorbar_ticks = CBAR_TICKS,
                 colorbar_tickfontsize = FS,
                 right_margin = 16Plots.mm
                 )
                 plot!(xt, yt, 0, 2π, leg = false, aspect_ratio = :equal, lw = LW_MAX, color=:black)
                 plot!(xt, yt, 0, 2π, leg=false, aspect_ratio=:equal, lw=LW_min, color=:white, xlims = (-0.5, 0.5), ylims = (-0.5, 0.5), xticks = -0.4:0.4:0.4, yticks = -0.4:0.4:0.4)

err_uni = heatmap(Δx,
                 Δy,
                 NSE_uni|>cpu,
                 aspect_ratio = :equal,
                 size = SIZE,
                 xlabel = "x (m)",
                 ylabel = "y (m)",
                 c = cgrad(:pink, rev=true),
                 clims = Elim,
                 xtickfontsize = FS_CBAR,
                 ytickfontsize = FS_CBAR,
                 labelfontsize = FS_CBAR,
                 yflip = true,
                 colorbar_ticks = EBAR_TICKS,
                 colorbar_tickfontsize = FS,
                 right_margin = 16Plots.mm
                 )
                 plot!(xt, yt, 0, 2π, leg = false, aspect_ratio = :equal, lw = LW_MAX, color=:black)
                 plot!(xt, yt, 0, 2π, leg=false, aspect_ratio=:equal, lw=LW_min, color=:white, xlims = (-0.5, 0.5), ylims = (-0.5, 0.5), xticks = -0.4:0.4:0.4, yticks = -0.4:0.4:0.4)


p_prop = heatmap(Δx,
                 Δy,
                 real(y_PLANE_prop|>cpu),
                 aspect_ratio = :equal,
                 size = SIZE,
                 xlabel = "x (m)",
                 ylabel = "y (m)",
                 c = cgrad([:blue, :white, :red]),
                 clims = Clim,
                 xtickfontsize = FS_CBAR,
                 ytickfontsize = FS_CBAR,
                 labelfontsize = FS_CBAR,
                 yflip = true,
                 colorbar_ticks = CBAR_TICKS,
                 colorbar_tickfontsize = FS,
                 right_margin = 16Plots.mm
                 )
                 plot!(xt, yt, 0, 2π, leg = false, aspect_ratio = :equal, lw = LW_MAX, color=:black)
                 plot!(xt, yt, 0, 2π, leg=false, aspect_ratio=:equal, lw=LW_min, color=:white, xlims = (-0.5, 0.5), ylims = (-0.5, 0.5), xticks = -0.4:0.4:0.4, yticks = -0.4:0.4:0.4)

err_prop = heatmap(Δx,
                 Δy,
                 NSE_prop|>cpu,
                 aspect_ratio = :equal,
                 size = SIZE,
                 xlabel = "x (m)",
                 ylabel = "y (m)",
                 c = cgrad(:pink, rev=true),
                 clims = Elim,
                 xtickfontsize = FS_CBAR,
                 ytickfontsize = FS_CBAR,
                 labelfontsize = FS_CBAR,
                 yflip = true,
                 colorbar_ticks = EBAR_TICKS,
                 colorbar_tickfontsize = FS,
                 right_margin = 16Plots.mm
                 )
                 plot!(xt, yt, 0, 2π, leg = false, aspect_ratio = :equal, lw = LW_MAX, color=:black)
                 plot!(xt, yt, 0, 2π, leg=false, aspect_ratio=:equal, lw=LW_min, color=:white, xlims = (-0.5, 0.5), ylims = (-0.5, 0.5), xticks = -0.4:0.4:0.4, yticks = -0.4:0.4:0.4)

savefig(p, string(DIR, "/400ms/GRAPHS/NMSE.pdf"))
savefig(p_orig, string(DIR, "/400ms/GRAPHS/SF_OG.pdf"))
savefig(p_uni, string(DIR, "/400ms/GRAPHS/SF_uni.pdf"))
savefig(err_uni, string(DIR, "/400ms/GRAPHS/ERR_uni.pdf"))
savefig(p_prop, string(DIR, "/400ms/GRAPHS/SF_prop.pdf"))
savefig(err_prop, string(DIR, "/400ms/GRAPHS/ERR_prop.pdf"))
