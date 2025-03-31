# Contribution guide

We welcome and encourage contributions to Ket.
This guide explains some ways to contribute.

!!! tip
    Questions, suggestions, bugs and so forth can be added [as an issue on the repository](https://github.com/dev-ket/Ket.jl/issues). You can also find us on [Slack](https://join.slack.com/t/ketjl/shared_invite/zt-32tuwt7qb-fQ4sqfpVID9E_dzMQRrnmA).

## Contributing code

The basic guidelines for contributing code are:

- The problems to be solved should be of wide and recurrent use, but not trivial.
- Functions must have [docstrings](https://docs.julialang.org/en/v1/manual/documentation/) explaining the inputs and usage.
- Every function should come with [automated tests](https://docs.julialang.org/en/v1/stdlib/Test/) (these tests are automatically run on every commit to verify nothing is broken).
- Try to [write fast code](https://docs.julialang.org/en/v1/manual/performance-tips/), for example by enforcing type stability and reusing existing, optimized code.
- Use [generic typing](https://docs.julialang.org/en/v1/manual/style-guide/#Avoid-writing-overly-specific-types) to enable arbitrary precision computations.
- Minimize dependencies on external packages.

There are only three steps to add a new function to Ket:

- Write the function and the docstring in the appropriate file of the [source code](https://github.com/dev-ket/Ket.jl/tree/master/src), maintaining the same code formatting style.
- Think of some tests to guarantee that your function is working as intended and add them to the [test files](https://github.com/dev-ket/Ket.jl/tree/master/test).
- Include your function's name in the [list of functions](https://github.com/dev-ket/Ket.jl/blob/master/docs/src/api.md) so that it appears in the documentation.

You should then do a pull request to the repository, and all the automated tests will run. It is a good idea to run the tests locally before committing to the repository.

!!! tip
    This procedure can sound complicated if you do not have experience with Git and Julia development. Checking [the JuMP guide](https://jump.dev/JuMP.jl/dev/developers/contributing/#Contribute-code-to-JuMP) or [Modern Julia Workflows](https://modernjuliaworkflows.org/) may help, otherwise reach out for someone to walk you through the process.

If you have an idea for a contribution but are unsure if it is useful, consider proposing it beforehand.
Otherwise, if you want to contribute but have no ideas, there is a [TODO](https://github.com/dev-ket/Ket.jl/blob/master/TODO) list.
You can also reach out to us if you need help optimizing your code or using generic types.

## Documentation

Another helpful way of contributing to Ket is to improve the documentation with new examples or improved docstrings.

### Contributing examples

We encourage submissions of examples that build on Ket.
Doing this is very simple: You just have to write a `.jl` file that may contain [Markdown](https://docs.julialang.org/en/v1/stdlib/Markdown/) (text, equations etc.) and Julia code, then add it to the [Ket.jl/docs/src/examples](https://github.com/dev-ket/Ket.jl/tree/master/docs/src/examples) directory.

When this is committed to the repository, the example files will be automatically processed and displayed in the documentation. 
You can follow the syntax from the existing examples, or check out the documentation for [Literate.jl](https://fredrikekre.github.io/Literate.jl/v2/).

### Docstrings

The [list of functions](https://dev-ket.github.io/Ket.jl/dev/api/) is the most important part of the documentation, and its purpose is to explain how each function works, including the input arguments and the expected results.
This list is generated automatically from the ["docstrings"](https://docs.julialang.org/en/v1/manual/documentation/) that precede each function in the source code. 

If you find something that can be clarified or extended, you can suggest or submit modifications to the docstrings.

## Bug reports

Whenever you notice crashing code, incorrect implementations, or performance issues, please [open an issue on the repository](https://github.com/dev-ket/Ket.jl/issues).
Include as much information about the problem as you can, and ideally provide a minimal working example of code where the problem appears.
In case you understand the issue, you can also suggest a fix or consider submitting a patch.
