"""
    white_noise(ρ::AbstractMatrix, v::Real)

Returns `v * ρ + (1 - v) * id`, where `id` is the maximally mixed state.
"""
function white_noise(ρ::AbstractMatrix, v::Real)
    return white_noise!(copy(ρ), v)
end
export white_noise

"""
    white_noise!(ρ::AbstractMatrix, v::Real)

Modifies `ρ` in place to tranform it into `v * rho + (1 - v) * id`
where `id` is the maximally mixed state.
"""
function white_noise!(ρ::AbstractMatrix, v::Real)
    v == 1 && return ρ
    parent(ρ) .*= v
    tmp = (1 - v) / size(ρ, 1)
    # https://discourse.julialang.org/t/change-the-diagonal-of-an-abstractmatrix-in-place/67294/2
    for i ∈ axes(ρ, 1)
        @inbounds ρ[i, i] += tmp
    end
    return ρ
end
export white_noise!

"""
    state_bell_ket([T=ComplexF64,] a::Integer, b::Integer, d::Integer = 2)

Produces the ket of the generalized Bell state ψ_`ab` of local dimension `d`.
"""
function state_bell_ket(::Type{T}, a::Integer, b::Integer, d::Integer = 2; coeff = inv(_sqrt(T, d))) where {T<:Number}
    ψ = zeros(T, d^2)
    ω = _root_unity(T, d)
    val = T(0)
    for i ∈ 0:d-1
        ψ[d*i+mod(a + i, d)+1] = coeff * _phase(ω, i * b, d)
    end
    return ψ
end
state_bell_ket(a, b, d::Integer = 2) = state_bell_ket(ComplexF64, a, b, d)
export state_bell_ket

"""
    state_bell([T=ComplexF64,] a::Integer, b::Integer, d::Integer = 2, v::Real = 1)

Produces the generalized Bell state ψ_`ab` of local dimension `d` with visibility `v`.
"""
function state_bell(::Type{T}, a::Integer, b::Integer, d::Integer = 2, v::Real = 1) where {T<:Number}
    ρ = ketbra(state_bell_ket(T, a, b, d; coeff = one(T)))
    parent(ρ) ./= d
    return white_noise!(ρ, v)
end
state_bell(a, b, d::Integer = 2) = state_bell(ComplexF64, a, b, d)
export state_bell

"""
    state_phiplus_ket([T=ComplexF64,] d::Integer = 2)

Produces the ket of the maximally entangled state ϕ⁺ of local dimension `d`.
"""
function state_phiplus_ket(::Type{T}, d::Integer = 2; kwargs...) where {T<:Number}
    return state_ghz_ket(T, 2, d; kwargs...)
end
state_phiplus_ket(d::Integer = 2; kwargs...) = state_phiplus_ket(ComplexF64, d; kwargs...)
export state_phiplus_ket

"""
    state_phiplus([T=ComplexF64,] d::Integer = 2; v::Real = 1)

Produces the maximally entangled state ϕ⁺ of local dimension `d` with visibility `v`.
"""
function state_phiplus(::Type{T}, d::Integer = 2; v::Real = 1) where {T<:Number}
    rho = ketbra(state_phiplus_ket(T, d; coeff = one(T)))
    parent(rho) ./= d
    return white_noise!(rho, v)
end
state_phiplus(d::Integer = 2; v::Real = 1) = state_phiplus(ComplexF64, d; v)
export state_phiplus

"""
    state_psiminus_ket([T=ComplexF64,] d::Integer = 2)

Produces the ket of the maximally entangled state ψ⁻ of local dimension `d`.
"""
function state_psiminus_ket(::Type{T}, d::Integer = 2; coeff = inv(_sqrt(T, d))) where {T<:Number}
    psi = zeros(T, d^2)
    psi[d.+(d-1)*(0:d-1)] .= (-1) .^ (0:d-1) .* coeff
    return psi
end
state_psiminus_ket(d::Integer = 2; kwargs...) = state_psiminus_ket(ComplexF64, d; kwargs...)
export state_psiminus_ket

"""
    state_psiminus([T=ComplexF64,] d::Integer = 2; v::Real = 1)

Produces the maximally entangled state ψ⁻ of local dimension `d` with visibility `v`.
"""
function state_psiminus(::Type{T}, d::Integer = 2; v::Real = 1) where {T<:Number}
    rho = ketbra(state_psiminus_ket(T, d; coeff = one(T)))
    parent(rho) ./= d
    return white_noise!(rho, v)
end
state_psiminus(d::Integer = 2; v::Real = 1) = state_psiminus(ComplexF64, d; v)
export state_psiminus

"""
    state_ghz_ket([T=ComplexF64,] N::Integer = 3, d::Integer = 2; coeff = 1/√d)

Produces the ket of the GHZ state with `N` parties and local dimension `d`.
"""
function state_ghz_ket(::Type{T}, N::Integer = 3, d::Integer = 2; coeff = inv(_sqrt(T, d))) where {T<:Number}
    psi = zeros(T, d^N)
    spacing = (1 - d^N) ÷ (1 - d)
    psi[1:spacing:d^N] .= coeff
    return psi
end
state_ghz_ket(N::Integer = 3, d::Integer = 2; kwargs...) = state_ghz_ket(ComplexF64, N, d; kwargs...)
export state_ghz_ket

"""
    state_ghz([T=ComplexF64,] N::Integer = 3, d::Integer = 2; v::Real = 1, coeff = 1/√d)

Produces the GHZ state with `N` parties, local dimension `d`, and visibility `v`.
"""
function state_ghz(::Type{T}, N::Integer = 3, d::Integer = 2; v::Real = 1, kwargs...) where {T<:Number}
    return white_noise!(ketbra(state_ghz_ket(T, N, d; kwargs...)), v)
end
state_ghz(N::Integer = 3, d::Integer = 2; kwargs...) = state_ghz(ComplexF64, N, d; kwargs...)
export state_ghz

"""
    state_w_ket([T=ComplexF64,] N::Integer = 3; coeff = 1/√N)

Produces the ket of the `N`-partite W state.
"""
function state_w_ket(::Type{T}, N::Integer = 3; coeff = inv(_sqrt(T, N))) where {T<:Number}
    psi = zeros(T, 2^N)
    psi[2 .^ (0:N-1).+1] .= coeff
    return psi
end
state_w_ket(N::Integer = 3; kwargs...) = state_w_ket(ComplexF64, N; kwargs...)
export state_w_ket

"""
    state_w([T=ComplexF64,] N::Integer = 3; v::Real = 1, coeff = 1/√N)

Produces the `N`-partite W state with visibility `v`.
"""
function state_w(::Type{T}, N::Integer = 3; v::Real = 1, kwargs...) where {T<:Number}
    return white_noise!(ketbra(state_w_ket(T, N; kwargs...)), v)
end
state_w(N::Integer = 3; kwargs...) = state_w(ComplexF64, N; kwargs...)
export state_w

"""
    state_supersinglet_ket([T=ComplexF64,] N::Integer = 3; coeff = 1/√N!)

Produces the ket of the `N`-partite `N`-level singlet state.

Reference: Adán Cabello, [arXiv:quant-ph/0203119](https://arxiv.org/abs/quant-ph/0203119)
"""
function state_supersinglet_ket(::Type{T}, N::Integer = 3; coeff = inv(_sqrt(T, factorial(N)))) where {T<:Number}
    psi = zeros(T, N^N)
    for per ∈ Combinatorics.permutations(1:N)
        tmp = kron(ket.((T,), per, (N,))...)
        if Combinatorics.parity(per) == 0
            # SD: this syntax allocates quite a bit
            # SD: we could go for an explicit one like for GHZ
            psi .+= tmp
        else
            # SD: this creates -0.0 in the output
            psi .-= tmp
        end
    end
    psi .*= coeff
    return psi
end
state_supersinglet_ket(N::Integer = 3; kwargs...) = state_supersinglet_ket(ComplexF64, N; kwargs...)
export state_supersinglet_ket

"""
    state_supersinglet([T=ComplexF64,] N::Integer = 3; v::Real = 1)

Produces the `N`-partite `N`-level singlet state with visibility `v`.
This state is invariant under simultaneous rotations on all parties: ``(U ⊗ ... ⊗ U) ρ (U ⊗ ... ⊗ U)' = ρ``.

Reference: Adán Cabello, [arXiv:quant-ph/0203119](https://arxiv.org/abs/quant-ph/0203119)
"""
function state_supersinglet(::Type{T}, N::Integer = 3; v::Real = 1) where {T<:Number}
    rho = ketbra(state_supersinglet_ket(T, N; coeff = one(T)))
    parent(rho) ./= factorial(N)
    return white_noise!(rho, v)
end
state_supersinglet(N::Integer = 3; kwargs...) = state_supersinglet(ComplexF64, N; kwargs...)
export state_supersinglet

"""
    state_dicke_ket([T=ComplexF64,] N::Integer, k::Integer; coeff = 1/√Cᴺₖ)

Produces the ket of the `N`-partite Dicke state with `k` excitations.

Reference: Robert H. Dicke [doi:10.1103/PhysRev.93.99](https://doi.org/10.1103/PhysRev.93.99)
"""
function state_dicke_ket(::Type{T}, N::Integer, k::Integer; coeff = inv(_sqrt(T, binomial(N, k)))) where {T<:Number}
    N > 0 && 0 ≤ k ≤ N || throw(ArgumentError("Invalid number of excitations"))
    psi = zeros(T, 2^N)
    ind = zeros(Int8, N)
    @inbounds for i ∈ eachindex(psi)
        if sum(ind) == k
            psi[i] = coeff
        end
        _update_odometer!(ind, 2)
    end
    return psi
end
state_dicke_ket(N::Integer, k::Integer; kwargs...) = state_dicke_ket(ComplexF64, N, k; kwargs...)
export state_dicke_ket

"""
    state_dicke([T=ComplexF64,] N::Integer, k::Integer; v::Real = 1)

Produces the `N`-partite Dicke state with `k` excitations.

Reference: Robert H. Dicke [doi:10.1103/PhysRev.93.99](https://doi.org/10.1103/PhysRev.93.99)
"""
function state_dicke(::Type{T}, N::Integer, k::Integer; v::Real = 1) where {T<:Number}
    rho = ketbra(state_dicke_ket(T, N, k; coeff = one(T)))
    parent(rho) ./= binomial(N, k)
    return white_noise!(rho, v)
end
state_dicke(N::Integer, k::Integer; kwargs...) = state_dicke(ComplexF64, N, k; kwargs...)
export state_dicke

"""
    state_horodecki33([T=ComplexF64,] a::Real)

Produces the 3 × 3 bipartite PPT-entangled Horodecki state with parameter `a` ∈ [0,1], which is PPT-entangled for `a` ∈ (0,1).

Reference: Paweł Horodecki, [arXiv:quant-ph/9703004](https://arxiv.org/abs/quant-ph/9703004)
"""
function state_horodecki33(::Type{T}, a::Real; v::Real = 1) where {T<:Number}
    @assert 0 ≤ a ≤ 1 "Parameter `a` must be in [0, 1]"
    x = (1 + a) / 2
    y = sqrt(1 - a^2) / 2
    rho = T[
        a 0 0 0 a 0 0 0 a
        0 a 0 0 0 0 0 0 0
        0 0 a 0 0 0 0 0 0
        0 0 0 a 0 0 0 0 0
        a 0 0 0 a 0 0 0 a
        0 0 0 0 0 a 0 0 0
        0 0 0 0 0 0 x 0 y
        0 0 0 0 0 0 0 a 0
        a 0 0 0 a 0 y 0 x
    ]
    rho ./= 8a + 1
    return white_noise!(Hermitian(rho), v)
end
state_horodecki33(a::Real) = state_horodecki33(ComplexF64, a)
export state_horodecki33

"""
    state_horodecki24([T=ComplexF64,] b::Real)

Produces the 2 × 4 bipartite PPT-entangled Horodecki state with parameter `b` ∈ [0,1], which is PPT-entangled for `b` ∈ (0,1).

Reference: Paweł Horodecki, [arXiv:quant-ph/9703004](https://arxiv.org/abs/quant-ph/9703004)
"""
function state_horodecki24(::Type{T}, b::Real; v::Real = 1) where {T<:Number}
    @assert 0 ≤ b ≤ 1 "Parameter `b` must be in [0, 1]"
    x = (1 + b) / 2
    y = sqrt(1 - b^2) / 2
    rho = T[
        b 0 0 0 0 b 0 0
        0 b 0 0 0 0 b 0
        0 0 b 0 0 0 0 b
        0 0 0 b 0 0 0 0
        0 0 0 0 x 0 0 y
        b 0 0 0 0 b 0 0
        0 b 0 0 0 0 b 0
        0 0 b 0 y 0 0 x
    ]
    rho ./= 7b + 1
    return white_noise!(Hermitian(rho), v)
end
state_horodecki24(b::Real) = state_horodecki24(ComplexF64, b)
export state_horodecki24

"""
    state_grid([T=ComplexF64], dA::Integer, dB::Integer, edges::Vector{Vector{ntuple{2, Int}}}; weights::Vector{T} = ones(T, length(edges)))

Produces the bipartite `dA` × `dB` grid state according to the `dA` × `dB` 2D (hyper-)graph with `edges` and `weights`.

Reference:
- Lockhart et al., [arXiv:1705.09261](http://arxiv.org/abs/1705.09261)
- Ghimire et al., [arXiv:2207.09826](https://arxiv.org/abs/2207.09826)
"""
function state_grid(::Type{T}, edges::Vector{Vector{NTuple{2, Int}}}, dA::Integer = maximum(x -> x[1], Iterators.flatten(edges)), dB::Integer = maximum(x -> x[2], Iterators.flatten(edges)); weights::Vector{T} = ones(T, length(edges)), v::Real = 1) where {T<:Number}
    rho = zeros(T, dA * dB, dA * dB)
    for (i, e) ∈ enumerate(edges)
        edge_ket = zeros(T, dA * dB)
        for v ∈ e
            edge_ket += kron(ket(T, v[1], dA), ket(T, v[2], dB))
        end
        rho .+= weights[i] * ketbra(edge_ket)
    end
    rho ./= tr(rho)
    return white_noise!(Hermitian(rho), v)
end
state_grid(edges::Vector{Vector{NTuple{2, Int}}}, dA::Integer = maximum(x -> x[1], Iterators.flatten(edges)), dB::Integer = maximum(x -> x[2], Iterators.flatten(edges)); kwargs...) = state_grid(ComplexF64, edges, dA, dB; kwargs...)
export state_grid

"""
    state_crosshatch([T=ComplexF64])

Produces a bound entangled bipartite 3 × 3 crosshatch state.

Reference: Lockhart et al., [arXiv:1705.09261](http://arxiv.org/abs/1705.09261)
"""
function state_crosshatch(::Type{T}; v::Real = 1) where {T<:Number}
    dA, dB = 3, 3
    edges = [[(1, 1), (3, 2)], [(1, 2), (3, 3)], [(2, 1), (1, 3)], [(3, 1), (2, 3)]]
    return state_grid(T, edges, dA, dB; v)
end
state_crosshatch() = state_crosshatch(ComplexF64)
export state_crosshatch

"""
    state_sindici_piani_ket([T=ComplexF64], d::Integer, k::Integer = 1, l::Integer = d; coeff = inv(_sqrt(T, 2)))

Produces the ket of Sindici-Piani state of even local dimensions `d` × `d`

Reference: Sindici and Piani, [arXiv:1708.06595](http://arxiv.org/abs/1708.06595)
"""
function state_sindici_piani_ket(::Type{T}, d::Integer; coeffs = ones(T, d ÷ 2) .* inv(_sqrt(T, d ÷ 2))) where {T<:Number}
    @assert iseven(d) "The Sindici-Piani state is defined for even `d`."
    ψ = zeros(T, d^2)
    for i in 1:Int(d/2)
        _state_symmetric_ket!(ψ, d, 2i - 1, 2i; coeffs = [inv(_sqrt(T, 2)), -inv(_sqrt(T, 2))] .* coeffs[i])
    end
    return ψ
end
state_sindici_piani_ket(d::Integer) = state_sindici_piani_ket(ComplexF64, d)
export state_sindici_piani_ket

function _state_symmetric_ket!(ψ, d::Integer, k::Integer, l::Integer; coeffs = [inv(_sqrt(T, 2)), inv(_sqrt(T, 2))])
    ψ[d*(k-1)+l] = coeffs[1]
    ψ[d*(l-1)+k] = coeffs[2]
end

"""
    state_sindici_piani([T=ComplexF64], d::Integer; coeffs = [√2/√d, ⋯, √2/√d])

Produces the Sindici-Piani state of even local dimensions `d` × `d` with visibility `v`.

Reference: Sindici and Piani, [arXiv:1708.06595](http://arxiv.org/abs/1708.06595)
"""
function state_sindici_piani(::Type{T}, d::Integer; v::Real = 1) where {T<:Number}
    ρ = ketbra(state_sindici_piani_ket(T, d; coeffs = ones(T, d ÷ 2)))
    parent(ρ) ./= d / 2
    return white_noise!(ρ, v)
end
state_sindici_piani(d::Integer) = state_sindici_piani(ComplexF64, d)
export state_sindici_piani
