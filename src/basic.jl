"""
    ket([T=Bool,] i::Integer, d::Integer = 2)

Produces a ket of dimension `d` with nonzero element `i`.
"""
function ket(::Type{T}, i::Integer, d::Integer = 2) where {T<:Number}
    psi = zeros(T, d)
    psi[i] = 1
    return psi
end
ket(i::Integer, d::Integer = 2) = ket(Bool, i, d)
export ket

"""
    ketbra(v::AbstractVector)

Produces a ketbra of vector `v`.
"""
function ketbra(v::AbstractVector)
    return Hermitian(v * v')
end
export ketbra

"""
    proj([T=Bool,] i::Integer, d::Integer = 2)

Produces a projector onto the basis state `i` in dimension `d`.
"""
function proj(::Type{T}, i::Integer, d::Integer = 2) where {T<:Number}
    pvec = zeros(T, d)
    pvec[i] = 1
    return Diagonal(pvec)
end
proj(i::Integer, d::Integer = 2) = proj(Bool, i, d)
export proj

"""
    shift([T=ComplexF64,] d::Integer, p::Integer = 1)

Constructs the shift operator X of dimension `d` to the power `p`.

Reference: [Generalized Clifford algebra](https://en.wikipedia.org/wiki/Generalized_Clifford_algebra)
"""
function shift(::Type{T}, d::Integer, p::Integer = 1) where {T<:Number}
    X = zeros(T, d, d)
    for i ∈ 0:d-1
        X[(i+p)%d+1, i+1] = 1
    end
    return X
end
shift(d::Integer, p::Integer = 1) = shift(ComplexF64, d, p)
export shift

"""
    clock([T=ComplexF64,] d::Integer, q::Integer = 1)

Constructs the clock operator Z of dimension `d` to the power `q`.

Reference: [Generalized Clifford algebra](https://en.wikipedia.org/wiki/Generalized_Clifford_algebra)
"""
function clock(::Type{T}, d::Integer, q::Integer = 1) where {T<:Number}
    z = zeros(T, d)
    ω = _root_unity(T, d)
    for i ∈ 0:d-1
        z[i+1] = _phase(ω, i * q, d)
    end
    return Diagonal(z)
end
clock(d::Integer, p::Integer = 1) = clock(ComplexF64, d, p)
export clock

"""
    shiftclock(v::AbstractVector, p::Integer, q::Integer)

Produces X^`p` * Z^`q` * `v`, where X and Z are the shift and clock operators of dimension length(`v`).

Reference: [Generalized Clifford algebra](https://en.wikipedia.org/wiki/Generalized_Clifford_algebra)
"""
function shiftclock(v::AbstractVector{T}, p, q) where {T}
    d = length(v)
    w = Vector{T}(undef, d)
    ω = _root_unity(T, d)
    @inbounds for i ∈ 0:d-1
        w[(i+p)%d+1] = v[i+1] * _phase(ω, i * q, d)
    end
    return w
end
export shiftclock

function _phase(ω::T, exp::Integer, d::Integer) where {T}
    expmod = exp % d
    if expmod == 0
        phase = T(1)
    elseif 4expmod == d
        phase = T(im)
    elseif 2expmod == d
        phase = T(-1)
    elseif 4expmod == 3d
        phase = T(-im)
    else
        phase = ω^expmod
    end
    return phase
end

"""
    pauli([T=ComplexF64,], ind::Vector{<:Integer})

Constructs the Pauli matrices: 0 or "I" for the identity,
1 or "X" for the Pauli X operation, 2 or "Y" for the Pauli Y
operator, and 3 or "Z" for the Pauli Z operator.
Vectors of integers between 0 and 3 or strings of I, X, Y, Z
automatically generate Kronecker products of the corresponding
operators.
"""
function pauli(::Type{T}, i::Integer) where {T<:Number}
    return gellmann(T, i ÷ 2 + 1, i % 2 + 1, 2)
end
function pauli(::Type{T}, ind::Vector{<:Integer}) where {T<:Number}
    if length(ind) == 1
        return pauli(T, ind[1])
    else
        return kron([pauli(T, i) for i ∈ ind]...)
    end
end
function pauli(::Type{T}, str::String) where {T<:Number}
    ind = Int[]
    for c ∈ str
        if c ∈ ['I', 'i', '1']
            push!(ind, 0)
        elseif c ∈ ['X', 'x']
            push!(ind, 1)
        elseif c ∈ ['Y', 'y']
            push!(ind, 2)
        elseif c ∈ ['Z', 'z']
            push!(ind, 3)
        else
            @warn "Unknown character"
        end
    end
    return pauli(T, ind)
end
pauli(::Type{T}, c::Char) where {T<:Number} = pauli(T, string(c))
pauli(i::Integer) = pauli(ComplexF64, i)
pauli(ind::Vector{<:Integer}) = pauli(ComplexF64, ind)
pauli(str::String) = pauli(ComplexF64, str)
pauli(c::Char) = pauli(ComplexF64, c)
export pauli

"""
    gellmann([T=ComplexF64,], d::Integer = 3)

Constructs the set `G` of generalized Gell-Mann matrices in dimension `d` such that
`G₁ = I` and `Tr(GᵢGⱼ) = 2 δᵢⱼ`.

Reference: [Generalizations of Pauli matrices](https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices)
"""
function gellmann(::Type{T}, d::Integer = 3) where {T<:Number}
    return [gellmann(T, i, j, d) for j ∈ 1:d, i ∈ 1:d][:]
    # d=2, ρ = 1/2(σ0 + n*σ)
    # d=3, ρ = 1/3(I + √3 n*λ)
    #      ρ = 1/d(I + sqrt(2/(d*(d-1))) n*λ)
    # SD: the next line would be for a potential KetSparse extension
    # SD: I Haven't thought yet how to deal with this.
    # return [gellmann(T, k, j, d, sparse(zeros(Complex{T}, d, d))) for j ∈ 1:d, k ∈ 1:d][:]
end
gellmann(d::Integer = 3) = gellmann(ComplexF64, d)
export gellmann

"""
    gellmann([T=ComplexF64,], i::Integer, j::Integer, d::Integer = 3)

Constructs the set `i`,`j`th Gell-Mann matrix of dimension `d`.

Reference: [Generalizations of Pauli matrices](https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices)
"""
function gellmann(::Type{T}, i::Integer, j::Integer, d::Integer = 3) where {T<:Number}
    return gellmann!(Hermitian(zeros(T, d, d)), i, j, d)
end
gellmann(i::Integer, j::Integer, d::Integer = 3) = gellmann(ComplexF64, i, j, d)

"""
    gellmann!(res::AbstractMatrix{T}, i::Integer, j::Integer, d::Integer = 3)

In-place version of `gellmann`.
"""
function gellmann!(res::Hermitian{T}, i::Integer, j::Integer, d::Integer = 3) where {T<:Number}
    if i < j
        parent(res)[i, j] = 1
    elseif i > j
        parent(res)[j, i] = -im
    elseif i == 1
        for k ∈ 1:d
            res[k, k] = 1 # _sqrt(T, 2) / _sqrt(T, d) if we want a proper normalisation
        end
    elseif i == d
        tmp = _sqrt(T, 2) / _sqrt(T, d * (d - 1))
        for k ∈ 1:d-1
            res[k, k] = tmp
        end
        res[d, d] = -(d - 1) * tmp
    else
        gellmann!(res, i, j, d - 1)
    end
    return res
end
export gellmann!

"""
    cleanup!(M::AbstractArray{T}; tol = _eps(T))

Zeroes out real or imaginary parts of `M` that are smaller than `tol`.
"""
function cleanup!(M::AbstractArray{T}; tol = _eps(T)) where {T<:Number}
    pM = parent(M)
    if isbitstype(T)
        M2 = reinterpret(real(T), pM) #this is a no-op when T<:Real
        _cleanup!(M2; tol)
    else
        reM = real(pM)
        imM = imag(pM)
        _cleanup!(reM; tol)
        _cleanup!(imM; tol)
        pM .= Complex.(reM, imM)
    end
    return M
end
export cleanup!

function _cleanup!(M; tol)
    return M[abs.(M).<tol] .= 0
end

function _orthonormal_range_svd!(A::AbstractMatrix{T}; tol::Union{Real,Nothing} = nothing) where {T<:Number}
    dec = svd!(A)
    tol = isnothing(tol) ? maximum(dec.S) * _eps(T) * minimum(size(A)) : tol
    rank = sum(dec.S .> tol)
    return dec.U[:, 1:rank]
end

_orthonormal_range_svd(A::AbstractMatrix; tol::Union{Real,Nothing} = nothing) = _orthonormal_range_svd!(copy(A); tol)

function _orthonormal_range_qr(A::SA.AbstractSparseMatrix{T,M}; tol::Union{Real,Nothing} = nothing) where {T<:Number,M}
    dec = qr(A)
    tol = isnothing(tol) ? maximum(abs.(dec.R)) * _eps(T) : tol
    rank = sum(abs.(Diagonal(dec.R)) .> tol)
    return @view SA.sparse(dec.Q)[dec.rpivinv, 1:rank]
end

"""
    orthonormal_range(A::AbstractMatrix{T}; tol::T=nothing) where {T<:Number}

Orthonormal basis for the range of `A`. When `A` is sparse and `T` ∈ [`Float64`, `ComplexF64`, `Float32`, `ComplexF32`], uses a QR factorization and returns a sparse result,
otherwise uses an SVD and returns a dense matrix.
Tolerance `tol` is used to compute the rank and is automatically set if not provided.
"""
function orthonormal_range(A::SA.AbstractMatrix{T}; tol::Union{Real,Nothing} = nothing) where {T<:Number}
    if (T <: SA.CHOLMOD.VTypes) && SA.issparse(A)
        return _orthonormal_range_qr(A; tol)
    else
        return _orthonormal_range_svd(Matrix(A); tol)
    end
end
export orthonormal_range

"""
    symmetric_projector(dim::Integer, n::Integer)

Computes the projector onto the symmetric subspace of `n` copies of a `dim`-dimensional space.

Reference: [Watrous' book](https://cs.uwaterloo.ca/~watrous/TQI/), Sec. 7.1.1
"""
function symmetric_projector(::Type{T}, dim::Integer, n::Integer) where {T}
    P = SA.spzeros(T, dim^n, dim^n)
    perms = Combinatorics.permutations(1:n)
    for perm ∈ perms
        P .+= permutation_matrix(T, dim, perm)
    end
    P ./= length(perms)
    return P
end
export symmetric_projector
symmetric_projector(dim::Integer, n::Integer) = symmetric_projector(Float64, dim, n)

"""
    symmetric_isometry(dim::Integer, n::Integer)

Computes an isometry that encodes the symmetric subspace of `n` copies of a `dim`-dimensional space. Specifically, it maps a vector space of dimension `binomial(n + dim -1, dim -1)` onto the symmetric subspace of the symmetric subspace of the vector space of dimension `dim^n`.

Reference: [Watrous' book](https://cs.uwaterloo.ca/~watrous/TQI/), Sec. 7.1.1
"""
function symmetric_isometry(::Type{T}, dim::Integer, n::Integer; partial::Bool = true) where {T}
    P = symmetric_projector(T, dim, n)
    V = orthonormal_range(P)
    size(V, 2) != binomial(n + dim - 1, dim - 1) && throw(AssertionError("Rank computation failed"))
    return V
end
export symmetric_isometry
symmetric_isometry(dim::Integer, n::Integer) = symmetric_isometry(Float64, dim, n)

"""
    n_body_basis(
    n::Integer,
    n_parties::Integer;
    sb::AbstractVector{<:AbstractMatrix} = [pauli(1), pauli(2), pauli(3)],
    eye::AbstractMatrix = I(size(sb[1], 1))

Return the basis of `n` nontrivial operators acting on `n_parties`, by default using sparse Pauli matrices.

For example, `n_body_basis(2, 3)` generates all products of two Paulis and one identity, so
``{X ⊗ X ⊗ 1, X ⊗ 1 ⊗ X, ..., X ⊗ Y ⊗ 1, ..., 1 ⊗ Z ⊗ Z}``.

Instead of Paulis, a basis can be provided by the parameter `sb`, and the identity can be changed with `eye`.

This function returns a generator, which can then be used e.g. in for loops without fully allocating the
entire basis at once. If you need a vector, call `collect` on it.
"""
function n_body_basis(
    n::Integer,
    n_parties::Integer;
    sb::AbstractVector{<:AbstractMatrix} = SA.sparse.([pauli(1), pauli(2), pauli(3)]),
    eye::AbstractMatrix = SA.sparse(one(eltype(sb[1])) * I, size(sb[1]))
)
    (n ≥ 0 && n_parties ≥ 2) || throw(ArgumentError("Number of parties must be ≥ 2 and n ≥ 0."))
    n ≤ n_parties || throw(ArgumentError("Number of parties cannot be larger than n."))

    sb = [sb; [eye]]
    nb = length(sb) - 1
    idx_eye = length(sb)
    basis = (
        kron(sb[t]...) for p ∈ Combinatorics.with_replacement_combinations(1:nb, n) for
        t ∈ Combinatorics.multiset_permutations([p; repeat([idx_eye], n_parties - n)], n_parties)
    )
    return basis
end
export n_body_basis

_ispossemidef(λ::AbstractVector{T}) where {T<:Real} = all(λ .> -_rtol(T))
_ispossemidef(A::AbstractMatrix{T}) where {T<:Number} = isposdef(A + _rtol(T) * I)
