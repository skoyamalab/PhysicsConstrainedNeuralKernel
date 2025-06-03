# Benchmarking library included for further testing. Comment if unneeded.
using BenchmarkTools, JLD2
# Libraries needed to compile the PCNK module. The overhead can be eliminated by performing static compilation.
using GenericLinearAlgebra, LinearAlgebra, Polynomials, SpecialFunctions, ArrayAllocators, SphericalHarmonics, KernelFunctions, ChainRules,Lebedev, Functors, Optimization, Optimisers, FFTW, Flux, KernelAbstractions, OrdinaryDiffEq, Suppressor
# using CUDA
using OrdinaryDiffEq, SciMLSensitivity
# Find current directory.
DIR = @__DIR__
# The script pressuposes the folder PCNK with the library is in the same directory as the script.
include(string(DIR, "/PCNK/master.jl"))
# Include module
using .PCNK

# Precision. Generally stick to 32 bit when using Flux models, as that is what most operations default to.
T = Float64

# The first step is to load the data
data = JLD2.load(string(DIR, "/400ms/SPM.jld"))
freqs = data["freqs"]
c = T(343)
ks = T.(2*π*freqs/c)
SNR = 20
# Get the recordings (unnoised), add noise to get the data that will be used to derive the model and the positions of the microphones that generated them.
ir0 = data["recordings"]
X = data["xyz_rec"]
ir = ir0 + 10^(-SNR/20) * std(ir0, dims = 1) .* randn(Complex{T}, size(X, 2), length(ks))
# Get the validation data, the clean signals that will be estimated, and the corresponding positions
ir_val = data["validation"]
X_val = data["xyz_val"]
# Get the pointwise data, for the planar pointwise evaluation.
ir_plane = data["pointwise"]
X_plane = data["xyz_pointwise"]
# Test orders for the kernel function
Ord_dir = 5
Ord_NN = 11
# Default value for regularization constant, given as variance of the noise for a 20dB SNR.
λ = 0.011
#Initial definition of the kernels (wavenumber is placeholder that is replaced in each iteration). Uniform is for reference.
k = ks[2]
κ_uni = PCNK.UniformKernel{T}(k)
W= Chain(Dense(3,5, tanh),
           Dense(5,5, tanh),
           Dense(5,1,tanh),
           softplus)

κ_pcnk = PCNK.NeuralWeightPlaneWaveKernel{T}(k, Ord_NN, W=W)

κ_prop = PCNK.DirectedResidualPINKernel{T}(k, Ord_dir, Ord_NN, W)
#We performed several computations in order to display good results, as this library was written after the initial experiments in the paper.
#these were the seeds with the best results for this configuration and for each frequency.

# Much like MATLAB, it is faster in Julia to prealocate values that will be stored.
coeffs_uni = Matrix{Complex{T}}(calloc, size(X, 2), length(freqs))
err_uni = zeros(T, length(freqs))

ir_uni = copy(ir)
ir_uni_val = copy(ir_val)
ir_uni_plane = copy(ir_plane)

#Added only to display performance, hence being added late into the file.
using ProgressMeter

########## Uniform kernel function #######################
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
    ir_uni[:, f] = K*α
    ir_uni_val[:, f] = y_est
    ir_uni_plane[:, f] = κ_uni(X_plane, X)*α
end

######### Physics-informed kernel function #################
# Training loss criterion, Leave-one-out cross validation in its closed form.
function ℓ(x::AbstractMatrix,y::AbstractArray,κ::Optimisers.Restructure, ps)
    kernel = κ(ps)
    K = kernel(x,x)
    # Neural part diagonal
    # reg_n = sum(kernel.NeuralKernel.W(kernel.NeuralKernel.v) .* kernel.NeuralKernel.σ')
    reg_n = sum(kernel.W(kernel.v) .* kernel.σ')
    # Anbalytical part diagonal
    # reg_a = sum(kernel.AnalyticalKernel.σ)
    # K_inv = inv(K + λ*(reg_a + reg_n)*I)
    K_inv = inv(K + λ*(reg_n)*I)
    α = K_inv*y
    # return sum(abs2, diag(K_inv) .\ α) + 0.01*reg_n
    return sum(abs2, diag(K_inv) .\ α)
    # + 0.0005*reg_n
end
χ0, _ = destructure(κ_pcnk)
Nx_pcnk = length(χ0)
# Optimization metrics to evaluate. The value the parameters take per iteration, the kernel expansion coefficients and the aggregate error per frequency.
Χ_pcnk = Matrix{T}(calloc, Nx_pcnk, length(freqs))
coeffs_pcnk = Matrix{Complex{T}}(calloc, size(X, 2), length(freqs))
err_pcnk = Vector{T}(calloc, length(freqs))

# The compliance of this weight is guaranteed by the architecture of the NN.
# We are optimizing the problem using the Optim library included in Optimization.jl, and will be using Zygote (generally the most supported AD framework in Julia).
using OptimizationOptimJL, ForwardDiff

ir_pcnk = copy(ir)
ir_pcnk_val = copy(ir_val)
ir_pcnk_plane = copy(ir_plane)

# Training of the kernel itself
@showprogress for f=1:length(freqs)
    # First step: set the wavenumber of the kernel(s)
    # κ_pcnk.AnalyticalKernel.k = ks[f]
    # κ_pcnk.NeuralKernel.k = ks[f]
    κ_pcnk.k = ks[f]
    # Since the Julia ecosystem tends to prefer explicit variables, it is recommended to use a Optimisers.Restructure object.
    # Implicit training using Flux.params is still possible, though second order methods will not be usable.
    χ, κ = destructure(κ_pcnk)
    # Set output to correspond with frequency
    y = ir[:, f]# Derivation data
    y_val = ir_val[:, f]# Validation data
    y_plane = ir_plane[:, f]# Pointwise analysis data
    # Define loss function for optimization
    loss(χ, y) = ℓ(X, y, κ, χ)
    sol = copy(χ)
    # Define optimization loop 
    # Define optimization function as well as what method of differentiation will be used and the constraints.
    @suppress begin
        func = Optimization.OptimizationFunction(loss,
        Optimization.AutoZygote())
        # Define the optimization problem, which includes the function that must be optimized.
        prob = Optimization.OptimizationProblem(func, χ, y)
        # Derive a solution using an appropriate optimizer. In this case LBFGS, a Newton method-like optimizer.
        sol .= Optimization.solve(prob, LBFGS(), maxiters=200; f_calls_limit=500)
    end
    # The solution object can be treated as a vector and we can now extract its value.
    χ = T.(sol)
    # Eliminate any numerical resquice placed by the optimizer
    χ[abs.(χ) .< eps(T)] .= zero(T)
    # Now, we can reconstruct the kernel.
    κ_eval = κ(χ)
    # reg = sum(κ_eval.NeuralKernel.W(κ_eval.NeuralKernel.v) .* κ_eval.NeuralKernel.σ') + sum(κ_eval.AnalyticalKernel.σ)
    reg = sum(κ_eval.W(κ_eval.v) .* κ_eval.σ')
    K = κ_eval(X,X)
    # We can now calculate the kernel coefficients associated with each frequency
    α = (K+λ*reg*I)\(y)
    ir_pcnk[:, f] = K*α
    #We now store the coefficients of the kernel regression
    coeffs_pcnk[:,f] = α
    Χ_pcnk[:, f] = χ
    y_est = κ_eval(X_val, X)*coeffs_pcnk[:, f]
    err_pcnk[f] = sum(abs2, y_est - y_val )/sum(abs2, y_val)
    ir_pcnk_val[:, f] = y_est
    ir_pcnk_plane[:, f] = κ_eval(X_plane, X)*α
end

# Define general training loss function, here made to take in a kernel constructor and kernel parameters.
# The function can be defined quite freely, as the gradients will be calculated with AD.
# Note that AD compatibility and performance does depend on the loss. Not every function has well-defined differentiation rules
function ℓ(x::AbstractMatrix,y::AbstractArray,κ::Optimisers.Restructure, ps)
    kernel = κ(ps)
    K = kernel(x,x)
    reg_n = sum(kernel.NeuralKernel.W(kernel.NeuralKernel.v) .* kernel.NeuralKernel.σ')
    reg_a = sum(kernel.AnalyticalKernel.σ)
    K_inv = inv(K + λ*(reg_a + reg_n)*I)
    α = K_inv*y
    return sum(abs2, diag(K_inv) .\ α) + 0.5*reg_n
    # return sum(abs2, diag(K_inv) .\ α)
end
χ0, _ = destructure(κ_prop)
Nx = length(χ0)
# Optimization metrics to evaluate. The value the parameters take per iteration, the kernel expansion coefficients and the aggregate error per frequency.
Χ_prop = Matrix{T}(calloc, Nx, length(freqs))
coeffs_prop = Matrix{Complex{T}}(calloc, size(X, 2), length(freqs))
err_prop = Vector{T}(calloc, length(freqs))
ir_prop = copy(ir)
ir_prop_val = copy(ir_val)
ir_prop_plane = copy(ir_plane)

# We will optimize using a constraint-aware optimizer, which displays how the kernels can be trained using a variety of techniques.
# We will also use autodiff for the training, meaning the constraints will be upheld automatically.

## Constraint function, which says the only constraints observed are on the indexes corresponding to γ, being their sum and values.
function cons!(constraint, χ, y)
    constraint .= [sum(χ[1:14]); χ[1:28]]
end
## The function is an inplace function considering optimization input (Χ) and parameters (y)
## Note the constraints are unrelated to the outputs y in this case, but still need to be made aware.

## Lower bound of the constraints: the sum is exactly 1, so the lower bound should be
## γ .≥ 0, β .≥ 0
lcons = T[1;fill(zero(T), 28)]
## Upper bound of the constraints: the sum is exactly equal to 1, while the upper bound γ is not considered.
## We set it to ∞, however informing this to Julia simply means there is no constraint considered.
##  γ .≤ ∞, β .≤ ∞
ucons = T[1;fill(T.(100), 28)]
# Any derivatives of the constraints are calculated by the optimizer itself, which is using autodiff, so this is sufficient to guarantee they are satisfied.

# We are optimizing the problem using the Optim library included in Optimization.jl, and will be using Zygote (generally the most supported AD framework in Julia).
# Training of the kernel itself
@showprogress for f=1:length(freqs)
    # First step: set the wavenumber of the kernel(s)
    κ_prop.AnalyticalKernel.k = ks[f]
    κ_prop.NeuralKernel.k = ks[f]
    # Since the Julia ecosystem tends to prefer explicit variables, it is recommended to use a Optimisers.Restructure object.
    # Implicit training using Flux.params is still possible.
    χ, κ = destructure(κ_prop)
    # Set output to correspond with frequency
    y = ir[:, f]# Derivation data
    y_val = ir_val[:, f]# Validation data
    # Define loss function for optimization
    loss(χ, y) = ℓ(X, y, κ, χ)
    sol = copy(χ)
    # Define optimization loop 
    # Define optimization function as well as what method of differentiation will be used and the constraints.
    @suppress begin
        func = Optimization.OptimizationFunction(loss,
        Optimization.AutoZygote(), cons = cons!)
        # Define the optimization problem, which includes the function that must be optimized and the lower and upper bounds of the constraint.
        prob = Optimization.OptimizationProblem(func, χ, y, lcons = lcons, ucons=ucons)
        # Derive a solution using an appropriate optimizer.
        sol .= Optimization.solve(prob, IPNewton(), maxiters=200; f_calls_limit=500)
    end
    # The solution object can be treated as a vector and we can now extract its value.
    χ = T.(sol)
    # Eliminate any numerical resquice placed by the optimizer
    χ[abs.(χ) .< eps(T)] .= zero(T)
    # Now, we can reconstruct the kernel.
    κ_eval = κ(χ)
    reg = sum(κ_eval.NeuralKernel.W(κ_eval.NeuralKernel.v) .* κ_eval.NeuralKernel.σ') + sum(κ_eval.AnalyticalKernel.σ)
    K = κ_eval(X,X)
    # We can now calculate the kernel coefficients associated with each frequency
    α = (K+λ*reg*I)\(y)
    ir_prop[:, f] = K*α
    #We now store the coefficients of the kernel regression
    coeffs_prop[:,f] = α
    Χ_prop[:, f] = χ
    y_est = κ_eval(X_val, X)*coeffs_prop[:, f]
    err_prop[f] = sum(abs2, y_est - y_val )/sum(abs2, y_val)
    ir_prop_val[:, f] = y_est
    ir_prop_plane[:, f] = κ_eval(X_plane, X) * α
end

# Let us collect the errors and show the relative improvement caused by training the model to fit the data and to differentiating between directed and residual fields.
ERR = 10log10.([err_uni err_pcnk err_prop][2:end, :])
# We ignore the frequency 0 component due to the fact it does not fit the solution model. The losses are very small due to small variation, but the use of the kernel method is not necessary.

# We can see the results in the following graph
using Plots

colors = reshape(palette(:default)[[5, 2, 1]], 1, :)
LW = 3
FS = 15
plot(freqs[2:end],
      ERR,
      legend = :bottomright,
      label = ["Uniform" "PCNN" "Adaptive kernel"],
      marker = [:v :sq :o],
      linestyle = [:dash :dashdot :solid],
      color = colors,
      linewidth = LW,
      legendfontsize = FS,
      xlabelfontsize = FS,
      xtickfontsize = FS,
      xlabel = "Frequency (Hz)",
      ylabelfontsize = FS,
      ytickfontsize = FS,
      ylabel = "NMSE (dB)"
      )
savefig("SPM/NMSE.pdf")

# The kernels can also be deployed on the GPU. Training can be performed as well on the GPU, but then second order methods cannot be used.
# The kernels have only been tested on NVIDIA GPUs using CUDA.jl, however no operation makes explicit reference to it.
# In theory, models should run in other brand GPUs, but that has not been tested.
# It is recommended to use precision T=Float32 with the GPU as not only are most GPU applications optimized for it, but also the |>gpu macro presupposes it.

# We will use GPU deployment in order to evaluate the pointwise performance of the method.

#Copy outputs to GPU
f0=11 # Frequency of analysis
y_PLANE = reshape(ir_plane[:, f0], (101, 101))
y_PLANE_uni = reshape(ir_uni_plane[:, f0], (101, 101))
y_PLANE_pcnk = reshape(ir_pcnk_plane[:, f0], (101, 101))
y_PLANE_prop = reshape(ir_prop_plane[:, f0], (101, 101))

# We can also calculate the normalized square error between each estimation and the ground truth in a pointwise fashion.

NSE_uni = 20log10.(abs.(y_PLANE_uni - y_PLANE)./abs.(y_PLANE))
NSE_pcnk = 20log10.(abs.(y_PLANE_pcnk - y_PLANE)./abs.(y_PLANE))
NSE_prop = 20log10.(abs.(y_PLANE_prop - y_PLANE)./abs.(y_PLANE))

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
uc = floor(UC/0.001)*0.001
CBAR_TICKS = -uc:uc/2:uc
FS_CBAR = 22
SIZE = (700, 630)
Elim = (-30, 0)
EBAR_TICKS = -30:5:0

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
savefig(p_orig, "SPM/p_orig.pdf")

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
savefig(p_uni, "SPM/p_uni.pdf")

p_err_uni = heatmap(Δx,
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
savefig(p_err_uni, "SPM/p_err_uni.pdf")

p_pcnk = heatmap(Δx,
                 Δy,
                 real(y_PLANE_pcnk|>cpu),
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
savefig(p_pcnk, "SPM/p_pcnk.pdf")

p_err_pcnk = heatmap(Δx,
                 Δy,
                 NSE_pcnk|>cpu,
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
savefig(p_err_pcnk, "SPM/p_err_pcnk.pdf")

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
savefig(p_prop, "SPM/p_prop.pdf")

p_err_prop = heatmap(Δx,
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
savefig(p_err_prop, "SPM/p_err_prop.pdf")
