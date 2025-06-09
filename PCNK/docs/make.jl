using PCNK
using Documenter

DocMeta.setdocmeta!(PCNK, :DocTestSetup, :(using PCNK); recursive=true)

makedocs(;
    modules=[PCNK],
    authors="ribeiro-juliano <ribeiro-juliano@ieee.org>",
    sitename="PCNK.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
