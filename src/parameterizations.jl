"""
    parameterized_unitary(λ::AbstractMatrix{T})

Produces the unitary matrix parameterized by λ.

Reference: Spengler, Huber, Hiesmayr, [arXiv:1004.5252](https://arxiv.org/abs/1004.5252)
"""
function parameterized_unitary(λ::AbstractMatrix{T}) where {T<:Real}
    U = Matrix{Complex{T}}(undef, size(λ))
    return _parameterized_unitary!(U, λ)
end
export parameterized_unitary

function _parameterized_unitary!(U::Matrix{Complex{T}}, λ::AbstractMatrix{T}) where {T<:Real}
    d = checksquare(U)
    # set U to the identity (without allocating)
    @inbounds for i ∈ 1:d
        U[i, i] = 1
        for j ∈ i+1:d
            U[i, j] = 0
            U[j, i] = 0
        end
    end
    # Eq. (1) from arXiv:1004.5252
    @inbounds for m ∈ 1:d-1, n ∈ m+1:d
        # exp(iPₙλₙₘ) where Pₙ=|n⟩⟨n|
        s, c = sincos(λ[n, m])
        e = Complex(c, s)
        for i ∈ 1:d
            U[i, n] *= e
        end
        # exp(iσₘₙλₘₙ) where σₘₙ=-i|m⟩⟨n|+i|n⟩⟨m|
        s, c = sincos(λ[m, n])
        for i ∈ 1:d
            uim = U[i, m]
            uin = U[i, n]
            U[i, m] = uim * c - uin * s
            U[i, n] = uim * s + uin * c
        end
    end
    @inbounds for l ∈ 1:d
        # exp(iPₗλₗₗ) where Pₗ=|l⟩⟨l|
        s, c = sincos(λ[l, l])
        e = Complex(c, s)
        for i ∈ 1:d
            U[i, l] *= e
        end
    end
    return U
end
