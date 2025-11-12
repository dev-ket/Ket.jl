#=
# Julia

Ket is a Julia package, and to use it Julia must be installed.
You can do that by following the instructions for your operating system in the [official Julia page](https://julialang.org/downloads/).

Julia can be used in different ways:

1. As a script interpreter, by running `julia script.jl` in the terminal, where `script.jl` is a file containing Julia code.

2. As a [Jupyter notebook](https://github.com/JuliaLang/IJulia.jl), which resembles the Mathematica interface, and where code blocks can be mixed with text and equations.

3. As a REPL (Read-Eval-Print Loop), which is a command-line interface where you can type Julia code and get immediate feedback, similar to MATLAB and the Python interpreter.

To access the REPL, you can run `julia` in the terminal. You will see the `julia>` prompt, and you can start coding right away (you can exit the REPL with `exit()` or pressing `Ctrl+D`).

If you are new to Julia, we recommend you to check out some [basic Julia tutorials](https://julialang.org/learning/) such as [this one](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_julia/#Getting-started-with-Julia) before proceeding.

# Installing Ket

Ket can be installed by running the following command in the Julia REPL:

```julia
] add Ket
```

The `]` key opens Julia's built-in package manager, and the `add` command installs the package.
Any officially registered package can be installed in the same way.

=#

# !!! tip
#     If you like living dangerously, you can instead install the development version by running `] add Ket#master`. There is usually no benefit in doing so, as we try to make a new release whenever the codebase is in a sensible state.

#=
After the installation is complete, you can check that Ket is working by running the following command in the REPL:
=#

using Ket
ket(1,2)

#=
The `using Ket` command tells Julia to load the Ket package, and that make its functions available in the current session.

# Using Ket

Ket is a package for quantum information theory, and it provides a set of tools for working with quantum states, measurements, and channels.
It is designed to be user-friendly and to provide a high-level interface for common tasks in quantum information theory.

The best way to learn how to use Ket is by using the [list of functions](https://dev-ket.github.io/Ket.jl/dev/api/) as a reference.
It is designed to be self-contained and to provide examples for each function.
Inside the REPL, you can also type `?` followed by the name of the function to get a brief description of its usage, or use the function `@doc`, for example:
=#

# ```julia
# @doc schmidt_decomposition
# ```
print(@doc schmidt_decomposition) #hide

#=
Beginners can be daunted by the first line in the function definitions, such as above, but they are only specifying the possible input arguments in terms of the types of variables.
Julia's type system is very powerful, but to use Ket you only need to know a few facts.

First, each variable in a function's input has a name, and the `::` symbol can be used to specify the type of the variable.
In `schmidt_number`, the input `ψ::AbstractVector` can be any vector, while `dims::AbstractVector{<:Integer}` is a vector of integers.

Some other functions accept an optional argument to specify the return type. For example:
=#

# ```julia
# @doc random_state
# ```
print(@doc random_state) #hide

#=
Here, the first argument `[T=ComplexF64,]` specifies that the return type of the function can be controlled by the user, and the default is `ComplexF64`.
This can be used to control the precision of the computations, but it is not necessary to know about it to use the function: You can just ignore it, and a sensible default type will be used.
Any argument followed by a "`=`" is optional, and the default value is specified after the "`=`" symbol (for example, `d::Integer = 2` above).

Finally, some functions also have *keyword* arguments.
They are specified after a `;` in the function definition:
=#

# ```julia
# @doc entanglement_robustness
# ```
print(@doc entanglement_robustness) #hide

#=
These are not *positional* arguments such as the ones that come before the `;`.
Instead, they must be passed by their names.
So, for example, `entanglement_robustness(ρ; noise = "separable")` is a valid call for any matrix ρ:
=#

λ, W = entanglement_robustness(state_ghz(2, 2); noise = "separable")
λ #hide
#-
W #hide

# !!! tip
#     Julia's REPL has a tab-completion feature that can be used to explore the functions available in Ket.
#     You can type `Ket.` and press `Tab` to see a list of functions, or start the name of a function (e.g., `state_`) and press `Tab` to see a list of available states.

#=
# Further resources

Knowing how to use the documentation should enable you to start adding the functions provided by Ket to your code right away.
If you are still unsure of what to do, or not yet convinced that Ket is useful for you, you may check some of the examples on the sidebar.

Other than learning the basics of Julia, we also recommend you to check the [JuMP tutorial](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_JuMP/) to learn how to use Julia for optimization problems.
This is one of the most popular uses of Julia, and it is very easy to use it with Ket to solve quantum information problems.
Ket integrates well with JuMP, offering for example functions that add common constraints to user-defined optimization problems, such as [`Ket._dps_constraints!`](@ref) and [`Ket._inner_dps_constraints!`](@ref).

Finally, if you have any questions, suggestions, bug reports, or wish to contribute with code or examples, you can reach out to us or open an issue in the [Ket repository](https://github.com/dev-ket/Ket.jl).
=#
