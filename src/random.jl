#often we don't actually need the variance of the normal variables to be 1, so we don't need to waste time diving everything by sqrt(2)
_randn(::Type{Complex{T}}) where {T<:AbstractFloat} = Complex{T}(randn(T), randn(T))
_randn(::Type{T}) where {T} = randn(T)
_randn(::Type{T}, dim1::Integer, dims::Integer...) where {T} = _randn!(Array{T}(undef, dim1, dims...))
function _randn!(A::AbstractArray{T}) where {T}
    for i ∈ eachindex(A)
        @inbounds A[i] = _randn(T)
    end
    return A
end

"""
    random_state([T=ComplexF64,] d::Integer, k::Integer = d)

Produces a uniformly distributed random quantum state in dimension `d` with rank `k`.

Reference: Życzkowski and Sommers, [arXiv:quant-ph/0012101](https://arxiv.org/abs/quant-ph/0012101)
"""
function random_state(::Type{T}, d::Integer, k::Integer = d) where {T}
    x = _randn(T, d, k)
    ρ = x * x'
    ρ ./= tr(ρ)
    return Hermitian(ρ)
end
random_state(d::Integer, k::Integer = d) = random_state(ComplexF64, d, k)
export random_state

"""
    random_state_ket([T=ComplexF64,] d::Integer)

Produces a Haar-random quantum state vector in dimension `d`.

Reference: Życzkowski and Sommers, [arXiv:quant-ph/0012101](https://arxiv.org/abs/quant-ph/0012101)
"""
function random_state_ket(::Type{T}, d::Integer) where {T}
    ψ = _randn(T, d)
    normalize!(ψ)
    return ψ
end
random_state_ket(d::Integer) = random_state_ket(ComplexF64, d)
export random_state_ket

#computes a Householder reflector with non-negative diagonal element
#according to the algorithm from http://www.netlib.org/lapack/lawnspdf/lawn203.pdf
#thanks to Seth Axen for the 3 functions below
#original source: https://gist.github.com/sethaxen/4d5a6d22e56794a90ece6224b8100f08
function _reflector!(v::AbstractVector)
    α = v[1]
    @views x = v[2:end]
    xnorm = norm(x)
    if iszero(xnorm) && isreal(α)
        τ = real(α) > 0 ? zero(α) : oftype(α, 2)
        β = abs(real(α))
        return τ
    end
    β = -copysign(hypot(α, xnorm), real(α))
    β, α, xnorm = _possibly_rescale!(β, α, x, xnorm)
    if β ≥ 0
        η = α - β
    else
        β = -β
        γ = real(α) + β
        if α isa Real
            η = -xnorm * (xnorm / γ)
        else
            imα = α - real(α)
            abs_imα = abs(imα)
            δ = -(abs_imα * (abs_imα / γ) + xnorm * (xnorm / γ))
            η = δ + imα
        end
    end
    v[1] = β
    x ./= η
    τ = -η / β
    return τ
end

_get_log2_safmin(::Type{T}) where {T<:Real} = exponent(floatmin(T) / eps(T))

function _possibly_rescale!(β, α, x, xnorm)
    T = float(real(eltype(x)))
    safmin_exponent = _get_log2_safmin(T)
    safmin = exp2(T(safmin_exponent))
    invsafmin = exp2(-T(safmin_exponent))
    if abs(β) ≥ safmin
        return β, α, xnorm
    end
    while abs(β) < safmin
        rmul!(x, invsafmin)
        β *= invsafmin
        α *= invsafmin
    end
    xnorm = norm(x)
    β = -copysign(hypot(α, xnorm), real(α))
    return β, α, xnorm
end

"""
    random_unitary([T=ComplexF64,] d::Integer)

Produces a Haar-random unitary matrix in dimension `d`.
If `T` is a real type the output is instead a Haar-random (real) orthogonal matrix.

References: Gilbert W. Stewart, [doi:10.1137/0717034](https://doi.org/10.1137/0717034)
            Demmel et al., [lawn203](http://www.netlib.org/lapack/lawnspdf/lawn203.pdf)
"""
random_unitary(::Type{T}, d::Integer) where {T<:Number} = _random_isometry(T, d, d)
random_unitary(d::Integer) = random_unitary(ComplexF64, d)
export random_unitary

function _random_isometry(::Type{T}, d::Integer, k::Integer) where {T<:Number}
    z = Matrix{T}(undef, d, k)
    @inbounds for j ∈ 1:k, i ∈ j:d
        z[i, j] = _randn(T)
    end
    τ = Vector{T}(undef, k)
    s = Vector{T}(undef, k)
    @inbounds for j ∈ 1:k #this is a partial QR decomposition where we don't apply the reflection to the rest of the matrix
        @views x = z[j:d, j]
        τ[j] = _reflector!(x)
    end
    return LinearAlgebra.QRPackedQ(z, τ)
end

"""
    random_isometry([T=ComplexF64,] d::Integer, k::Integer)

Produces a Haar-random isometry with `d` rows and `k` columns.

References: Gilbert W. Stewart, [doi:10.1137/0717034](https://doi.org/10.1137/0717034)
            Demmel et al., [lawn203](http://www.netlib.org/lapack/lawnspdf/lawn203.pdf)
"""
random_isometry(::Type{T}, d::Integer, k::Integer) where {T<:Number} = Matrix(_random_isometry(T, d, k))
#rather inefficient but until https://github.com/JuliaLang/LinearAlgebra.jl/issues/1172
#is solved this is the best we can do
random_isometry(d::Integer, k::Integer) = random_isometry(ComplexF64, d, k)
export random_isometry

"""
    random_povm([T=ComplexF64,] d::Integer, n::Integer, k::Integer)

Produces a random POVM of dimension `d` with `n` outcomes and rank `min(k, d)`.

Reference: Heinosaari et al., [arXiv:1902.04751](https://arxiv.org/abs/1902.04751)
"""
function random_povm(::Type{T}, d::Integer, n::Integer, k::Integer = d) where {T<:Number}
    d ≤ n * k || throw(ArgumentError("We need d ≤ n*k, but got d = $(d) and n*k = $(n * k)"))
    G = [randn(T, d, k) for _ ∈ 1:n]
    S = zeros(T, d, d)
    for i ∈ 1:n
        mul!(S, G[i], G[i]', true, true)
    end
    rootinvS = Hermitian(S)^-0.5 #don't worry, the probability of getting a singular S is zero
    E = [Matrix{T}(undef, d, d) for _ ∈ 1:n]
    temp = Matrix{T}(undef, d, k)
    for i ∈ 1:n
        mul!(temp, rootinvS, G[i])
        mul!(E[i], temp, temp')
    end
    return Hermitian.(E)
end
random_povm(d::Integer, n::Integer, k::Integer = d) = random_povm(ComplexF64, d, n, k)
export random_povm

"""
    random_probability([T=Float64,] d::Integer)

Produces a random probability vector of dimension `d` uniformly distributed on the simplex.

Reference: [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_variate_generation)
"""
function random_probability(::Type{T}, d::Integer) where {T}
    p = rand(T, d)
    p .= log.(p)
    #p .*= -1 not needed
    p ./= sum(p)
    return p
end
random_probability(d::Integer) = random_probability(Float64, d)
export random_probability
