_log(base::Real, x::Real) = x > 0 ? log(base, x) : zero(x)
_pow(x::Real, α::Real) = x > 0 ? x^α : zero(x)

"""
    relative_entropy(ρ::AbstractMatrix{T}, σ::AbstractMatrix{T}, α::real(T) = 1; base = 2)

Computes the (quantum) relative entropy tr(`ρ` (log `ρ` - log `σ`)) between positive semidefinite matrices `ρ` and `σ` using a base `base` logarithm.

If given `α` != 1 computes the (quantum) Rényi sandwiched relative entropy log(tr((`σ`^((1-`α`)/2`α`) * `ρ` * `σ`^((1-`α`)/2`α`))^`α`)) / (`α` - 1).

Note that the support of `ρ` must be contained in the support of `σ` but for efficiency this is not checked.

References:
- [Quantum relative entropy](https://en.wikipedia.org/wiki/Quantum_relative_entropy)
- [Rényi entropy](https://en.wikipedia.org/wiki/Rényi_entropy)
"""
function relative_entropy(
    ρ::AbstractMatrix{T},
    σ::AbstractMatrix{S},
    α::R = R(1);
    base = 2
) where {R<:Real,T<:Union{R,Complex{R}},S<:Union{R,Complex{R}}}
    d = checksquare(ρ)
    d == checksquare(σ) || throw(ArgumentError("ρ and σ must have the same size."))
    ρ_λ, ρ_U = eigen(ρ)
    σ_λ, σ_U = eigen(σ)
    if !_ispossemidef(ρ_λ) || !_ispossemidef(σ_λ)
        throw(DomainError("ρ and σ must be positive semidefinite"))
    end
    h = zero(α)
    if α == 1
        m = abs2.(ρ_U' * σ_U)
        logρ_λ = _log.(Ref(base), ρ_λ)
        logσ_λ = _log.(Ref(base), σ_λ)
        @inbounds for j ∈ 1:d, i ∈ 1:d
            h += ρ_λ[i] * (logρ_λ[i] - logσ_λ[j]) * m[i, j]
        end
    else
        rankρ = sum(ρ_λ .> _rtol(R))
        @views ρ_V = ρ_U[:, d-rankρ+1:end]
        @views ρ_μ = ρ_λ[d-rankρ+1:end]
        map!(x -> x < 0 ? R(0) : sqrt(x), ρ_μ, ρ_μ)
        map!(x -> _pow(x, (1 - α) / 2α), σ_λ, σ_λ)
        meat = Diagonal(σ_λ) * σ_U' * ρ_V * Diagonal(ρ_μ)
        ψ = svdvals(meat)
        ψ .= _pow.(ψ, Ref(2α))
        h = log(base, sum(ψ)) / (α - 1)
    end
    return h
end
export relative_entropy

"""
    relative_entropy(p::AbstractVector{T}, q::AbstractVector{T}, α::T = 1; base = 2)

Computes the relative entropy Σᵢ`p`ᵢlog(`p`ᵢ/`q`ᵢ) between two non-negative vectors `p` and `q` using a base `base` logarithm.

If given `α` != 1 computes the Rényi relative entropy log(Σᵢ`p`ᵢ^`α` * `q`ᵢ^(1-`α`)) / (`α` - 1).

Note that the support of `p` must be contained in the support of `q` but for efficiency this is not checked.

References:
- [Relative entropy](https://en.wikipedia.org/wiki/Relative_entropy)
- [Rényi entropy](https://en.wikipedia.org/wiki/Rényi_entropy)
"""
function relative_entropy(p::AbstractVector{T}, q::AbstractVector{T}, α::T = T(1); base = 2) where {T<:Real}
    if length(p) != length(q)
        throw(ArgumentError("p and q must have the same length."))
    end
    if any(p .< 0) || any(q .< 0)
        throw(DomainError("p and q must be non-negative"))
    end
    if α == 1
        h = zero(α)
        for i ∈ 1:length(p)
            if p[i] > 0
                h += p[i] * log(base, p[i] / q[i])
            end
        end
        return h
    else
        ψ = zero(α)
        for i ∈ 1:length(p)
            if p[i] > 0
                ψ += p[i]^α * q[i]^(1 - α)
            end
        end
        return log(base, ψ) / (α - 1)
    end
end

"""
    binary_relative_entropy(p::T, q::T, α::T = 1; base = 2)

Computes the binary relative entropy `p` log(`p`/`q`) + (1-`p`) log((1-`p`)/(1-`q`)) between two probabilities `p` and `q` using a base `base` logarithm.

If given `α` != 1 computes the Rényi binary relative entropy log(`p`^`α` * `q`^(1-`α`) + (1-`p`)^`α` * (1-`q`)^`α`) / (`α` - 1).

References:
- [Relative entropy](https://en.wikipedia.org/wiki/Relative_entropy)
- [Rényi entropy](https://en.wikipedia.org/wiki/Rényi_entropy)
"""
function binary_relative_entropy(p::T, q::T, α::T = T(1); base = 2) where {T<:Real}
    if p < 0 || p > 1 || q < 0 || q > 1
        throw(DomainError("p and q must be in [0, 1]"))
    end
    if p == q == 0 || p == q == 1
        return zero(float(p))
    end
    if (p == 0 && q == 1) || (p == 1 && q == 0)
        throw(DomainError("the support of [p, 1-p] must be contained in the support of [q, 1-q]"))
    end
    if α == 1
        return p * (_log(base, p) - _log(base, q)) + (1 - p) * (_log(base, 1 - p) - _log(base, 1 - q))
    else
        return log(base, p^α * q^(1 - α) + (1 - p)^α * (1 - q)^(1 - α)) / (α - 1)
    end
end
export binary_relative_entropy

"""
    entropy(ρ::AbstractMatrix{T}, α::real(T) = 1; base = 2)

Computes the von Neumann entropy -tr(`ρ` log `ρ`) of a positive semidefinite operator `ρ` using a base `base` logarithm.

If given `α` != 1 computes the Rényi entropy log(tr(`ρ`^`α`)) / (1 - `α`).

References:
- [von Neumann entropy](https://en.wikipedia.org/wiki/Von_Neumann_entropy)
- [Rényi entropy](https://en.wikipedia.org/wiki/Rényi_entropy)
"""
function entropy(ρ::AbstractMatrix{T}, α::R = R(1); base = 2) where {R<:Real,T<:Union{R,Complex{R}}}
    checksquare(ρ)
    λ = eigvals(ρ)
    _ispossemidef(λ) || throw(DomainError("ρ must be positive semidefinite"))
    if α == 1
        return -sum(λi * _log(base, λi) for λi ∈ λ)
    else
        return _log(base, sum(_pow(λi, α) for λi ∈ λ)) / (1 - α)
    end
end
export entropy

"""
    entropy(p::AbstractVector{T}, α::T = 1; base = 2)

Computes the Shannon entropy -Σᵢ`p`ᵢlog(`p`ᵢ) of a non-negative vector `p` using a base `base` logarithm.

If `α != 1` is given computes the Rényi entropy log(Σᵢ`p`ᵢ^`α` )/(1 - `α`).

References:
- [Entropy (information theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory))
- [Rényi entropy](https://en.wikipedia.org/wiki/Rényi_entropy)

"""
function entropy(p::AbstractVector{T}, α::T = T(1); base = 2) where {T<:Real}
    any(p .< 0) && throw(DomainError("p must be non-negative"))
    if α == 1
        return -sum(pi * _log(base, pi) for pi ∈ p)
    else
        return _log(base, sum(pi^α for pi ∈ p)) / (1 - α)
    end
end
export entropy

"""
    binary_entropy(p::T, α::T = T(1); base = 2)

Computes the Shannon entropy -`p` log(`p`) - (1-`p`)log(1-`p`) of a probability `p` using a base `base` logarithm.

If `α` != 1 is given computes the Rényi entropy log(`p`^`α` + (1-`p`)^`α`) / (1-`α`).

References:
- [Entropy (information theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory))
- [Rényi entropy](https://en.wikipedia.org/wiki/Rényi_entropy)
"""
function binary_entropy(p::T, α::T = T(1); base = 2) where {T<:Real}
    if p < 0 || p > 1
        throw(DomainError("p must be in [0, 1]"))
    end
    if p == 0 || p == 1
        return zero(float(p))
    end
    if α == 1
        return -p * log(base, p) - (1 - p) * log(base, 1 - p)
    else
        return log(base, p^α + (1 - p)^α) / (1 - α)
    end
end
export binary_entropy

"""
    conditional_entropy(pAB::AbstractMatrix{T}, α::T = 1; base = 2)

Computes the conditional Shannon entropy ∑ᵢⱼ`pAB`[i,j] * log(`pAB`[i|j]) of the joint probability distribution `pAB` using a base `base` logarithm.

If `α != 1` is given, computes the conditional Rényi entropy log(∑ⱼ`pAB`[j] * (∑ᵢ`pAB`[i|j]^α)^(1/α)) * α / (1 - α).

References:
- [Conditional entropy](https://en.wikipedia.org/wiki/Conditional_entropy)
- Müller-Lennert et al. [arXiv:1306.3142](https://arxiv.org/abs/1306.3142)
"""
function conditional_entropy(pAB::AbstractMatrix{T}, α::T = T(1); base = 2) where {T<:Real}
    nA, nB = size(pAB)
    if any(pAB .< 0)
        throw(DomainError("pAB must be non-negative"))
    end
    pB = sum(pAB; dims = 1)
    if α == 1
        h = zero(α)
        for b ∈ 1:nB
            if pB[b] > 0
                for a ∈ 1:nA
                    h -= pAB[a, b] * _log(base, pAB[a, b] / pB[b])
                end
            end
        end
    else
        ψ = zero(α)
        for b ∈ 1:nB
            if pB[b] > 0
                ψ += pB[b] * sum((pAB[a, b] / pB[b])^α for a ∈ 1:nA)^inv(α)
            end
        end
        h = log(base, ψ) * α / (1 - α)
    end
    return h
end

"""
    conditional_entropy(ρ::AbstractMatrix{T}, cond::Union{Integer,AbstractVector{<:Integer}}, dims::AbstractVecOrTuple, α::real(T) = 1; base = 2)

Computes the conditional von Neumann entropy of `ρ` with subsystem dimensions `dims` and conditioning systems `cond`, using a base `base` logarithm.

If `α != 1` is given, computes instead the following lower bound to the conditional Rényi entropy: -D`α`(`ρ`||I ⊗ `ρ`_`cond`). It is close to the true value when `α` is close to 1.

References:
- [Conditional quantum entropy](https://en.wikipedia.org/wiki/Conditional_quantum_entropy)
- Müller-Lennert et al. [arXiv:1306.3142](https://arxiv.org/abs/1306.3142)
"""
function conditional_entropy(
    ρ::AbstractMatrix{T},
    cond::Union{Integer,AbstractVector{<:Integer}},
    dims::AbstractVecOrTuple,
    α::R = R(1);
    base = 2
) where {R<:Real,T<:Union{R,Complex{R}}}
    isa(cond, Integer) && (cond = [cond])
    isempty(cond) && return entropy(ρ, α; base)
    length(cond) == length(dims) && return zero(real(eltype(ρ)))

    remove = Vector{eltype(cond)}(undef, length(dims) - length(cond))  # To condition on cond we trace out the rest
    counter = 0
    for i ∈ 1:length(dims)
        if !(i ∈ cond)
            counter += 1
            remove[counter] = i
        end
    end
    if α == 1
        ρ_cond = partial_trace(ρ, remove, dims)
        return entropy(ρ; base) - entropy(ρ_cond; base)
    else
        ρ_cond = trace_replace(ρ, remove, dims) * prod(dims[remove])
        return -relative_entropy(ρ, ρ_cond, α; base)
    end
end
export conditional_entropy
