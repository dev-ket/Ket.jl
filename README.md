[![Banner](https://dev-ket.github.io/Ket.jl/dev/assets/ket-jl-logo-dark-wide.svg)](https://dev-ket.github.io/Ket.jl/dev/)

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dev-ket.github.io/Ket.jl/dev/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14674642.svg)](https://doi.org/10.5281/zenodo.14674642)

Ket is a toolbox for quantum information, nonlocality, and entanglement written in the Julia programming language. All its functions are designed to work with generic types, allowing one to use `Int64` or `Float64` for efficiency, or arbitrary precision types when needed. Wherever possible they can be also used for optimization with [JuMP](https://jump.dev/JuMP.jl/stable/). And everything is optimized to the last microsecond.

Highlights:

* Work with multipartite Bell inequalities, computing their local bounds and Tsirelson bounds with `local_bound` and `tsirelson_bound`, and transforming between Collins-Gisin, probability, and correlation representations with `tensor_collinsgisin`, `tensor_probability`, and `tensor_correlation`.
* Work with bipartite entanglement by computing the relative entropy of entanglement, entanglement robustness, or Schmidt number via `entanglement_entropy`, `entanglement_robustness`, and `schmidt_number`. Under the hood these functions use the DPS hierarchy, which is also available in isolation via `_dps_constraints!`.
* Generate MUBs and SIC-POVMs through `mub` and `sic_povm`.
* Generate uniformly-distributed random states, unitaries, and POVMs with `random_state`, `random_unitary`, and `random_povm`.
* Generate well-known families of quantum states, such as the Bell states, the GHZ state, the W state, the Dicke states, and the super-singlet via `state_bell`, `state_ghz`, `state_w`, `state_dicke`, and `state_supersinglet`.
* Work with multilinear algebra via utility functions such as `partial_trace`, `partial_transpose`, and `permute_systems`.
* Generate kets with `ket`.

For the full list of functions see the [documentation](https://dev-ket.github.io/Ket.jl/dev/api/).

## Installation

Ket is a registered Julia package, so it can be installed by typing the following command in the Julia REPL:
```
]add Ket
```
This will install the latest released version. For the development version with the latest updates, use `]add Ket#master` instead.

## Contributing

We welcome suggestions, bug reports, documentation and code contributions.
See the [contributions guide](https://dev-ket.github.io/Ket.jl/dev/contribution_guide) for more information.

## Related libraries

Julia:

- [QuantumClifford.jl](https://github.com/QuantumSavory/QuantumClifford.jl)
- [QuantumOptics.jl](https://github.com/qojulia/QuantumOptics.jl)
- [QuantumToolbox.jl](https://github.com/qutip/QuantumToolbox.jl)
- [Yao.jl](https://github.com/QuantumBFS/Yao.jl)

Python:
- [toqito](https://github.com/vprusso/toqito)

MATLAB:
- [QETLAB](https://github.com/nathanieljohnston/QETLAB)
