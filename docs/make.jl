using Ket
using Documenter

DocMeta.setdocmeta!(Ket, :DocTestSetup, :(using Ket); recursive = true)

generated_path = joinpath(@__DIR__, "src")
base_url = "https://github.com/dev-ket/Ket.jl/blob/master/"
isdir(generated_path) || mkdir(generated_path)

open(joinpath(generated_path, "index.md"), "w") do io
    # Point to source license file
    println(
        io,
        """
        ```@meta
        EditURL = "$(base_url)README.md"
        ```
        """
    )
    # Write the contents out below the meta block
    for line ∈ eachline(joinpath(dirname(@__DIR__), "README.md"))
        println(io, line)
    end
end

makedocs(;
    modules = [Ket],
    authors = "Mateus Araújo et al.",
    repo = "https://github.com/dev-ket/Ket.jl/blob/{commit}{path}#{line}",
    sitename = "Ket.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://dev-ket.github.io/Ket.jl",
        edit_link = "master",
        assets = String[],
        sidebar_sitename = false
    ),
    pages = ["Home" => "index.md", "List of functions" => "api.md"],
    checkdocs = :exports
)

deploydocs(; repo = "github.com/dev-ket/Ket.jl", devbranch = "master", push_preview = true)

# makedocs(; sitename = "Ket", format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"))
