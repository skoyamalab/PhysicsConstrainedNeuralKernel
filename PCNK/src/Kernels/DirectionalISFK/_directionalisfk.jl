DISFK_src = @__DIR__

include(joinpath(DISFK_src, "Implementation.jl"))
include(joinpath(DISFK_src, "Evaluation.jl")) 
include(joinpath(DISFK_src, "Promotion.jl"))

export DirectionalISFKernel, FixedDirectionSFKernel, TrainableDirectionSFKernel