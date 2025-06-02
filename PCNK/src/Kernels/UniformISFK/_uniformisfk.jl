UISFK_src = @__DIR__

include(joinpath(UISFK_src, "Implementation.jl"))
include(joinpath(UISFK_src, "Promotion.jl"))
include(joinpath(UISFK_src, "Evaluation.jl"))

export UniformKernel