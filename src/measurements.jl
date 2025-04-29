_root_unity(::Type{T}, n::Integer) where {T<:Number} = exp(2 * im * real(T)(π) / n)
_sqrt(::Type{T}, n::Integer) where {T<:Number} = sqrt(real(T)(n))

# MUBs
function mub_prime(::Type{T}, p::Integer) where {T<:Number}
    γ = _root_unity(T, p)
    inv_sqrt_p = inv(_sqrt(T, p))
    B = [Matrix{T}(undef, p, p) for _ ∈ 1:p+1]
    B[1] .= I(p)
    if p == 2
        B[2] .= [1 1; 1 -1] .* inv_sqrt_p
        B[3] .= [1 1; im -im] .* inv_sqrt_p
    else
        for k ∈ 0:p-1
            fill!(B[k+2], inv_sqrt_p)
            for t ∈ 0:p-1, j ∈ 0:p-1
                exponent = mod(j * (t + k * j), p)
                if exponent == 0
                    continue
                elseif 4exponent == p
                    B[k+2][j+1, t+1] *= im
                elseif 2exponent == p
                    B[k+2][j+1, t+1] *= -1
                elseif 4exponent == 3p
                    B[k+2][j+1, t+1] *= -im
                else
                    B[k+2][j+1, t+1] *= γ^exponent
                end
            end
        end
    end
    return B
end
mub_prime(p::Integer) = mub_prime(ComplexF64, p)

function mub_prime_power(::Type{T}, p::Integer, r::Integer) where {T<:Number}
    d = p^r
    γ = _root_unity(T, p)
    inv_sqrt_d = inv(_sqrt(T, d))
    B = [zeros(T, d, d) for _ ∈ 1:d+1]
    B[1] .= I(d)
    f, x = Nemo.finite_field(p, r, "x")
    pow = [x^i for i ∈ 0:r-1]
    el = [sum(digits(i; base = p, pad = r) .* pow) for i ∈ 0:d-1]
    if p == 2
        for i ∈ 1:d, k ∈ 0:d-1, q ∈ 0:d-1
            aux = one(T)
            q_bin = digits(q; base = 2, pad = r)
            for m ∈ 0:r-1, n ∈ 0:r-1
                aux *= conj(im^_tr_ff(el[i] * el[q_bin[m+1]*2^m+1] * el[q_bin[n+1]*2^n+1]))
            end
            B[i+1][:, k+1] += (-1)^_tr_ff(el[q+1] * el[k+1]) * aux * B[1][:, q+1] * inv_sqrt_d
        end
    else
        inv_two = inv(2 * one(f))
        for i ∈ 1:d, k ∈ 0:d-1, q ∈ 0:d-1
            B[i+1][:, k+1] +=
                γ^_tr_ff(-el[q+1] * el[k+1]) * γ^_tr_ff(el[i] * el[q+1] * el[q+1] * inv_two) * B[1][:, q+1] * inv_sqrt_d
        end
    end
    return B
end
mub_prime_power(p::Integer, r::Integer) = mub_prime_power(ComplexF64, p, r)

# auxiliary function to compute the trace in finite fields as an Int
function _tr_ff(a::Nemo.FqFieldElem)
    Int(Nemo.lift(Nemo.ZZ, Nemo.absolute_tr(a)))
end

"""
    mub([T=ComplexF64,] d::Integer)

Construction of the standard complete set of MUBs.
The output contains 1+minᵢ pᵢ^rᵢ bases, where `d` = ∏ᵢ pᵢ^rᵢ.

Reference: Durt, Englert, Bengtsson, Życzkowski, [arXiv:1004.3348](https://arxiv.org/abs/1004.3348)
"""
function mub(::Type{T}, d::Integer) where {T<:Number}
    # the dimension d can be any integer greater than two
    @assert d ≥ 2
    f = collect(Nemo.factor(d))
    p = f[1][1]
    r = f[1][2]
    if length(f) > 1 # different prime factors
        B_aux1 = mub(T, p^r)
        B_aux2 = mub(T, d ÷ p^r)
        k = min(length(B_aux1), length(B_aux2))
        B = [Matrix{T}(undef, d, d) for _ ∈ 1:k]
        for j ∈ 1:k
            B[j] .= kron(B_aux1[j], B_aux2[j])
        end
    elseif r == 1 # prime
        return mub_prime(T, p)
    else # prime power
        return mub_prime_power(T, p, r)
    end
    return B
end
mub(d::Integer) = mub(ComplexF64, d)
export mub

# Select a specific subset with k bases
function mub(::Type{T}, d::Integer, k::Integer, s::Integer = 1) where {T<:Number}
    B = mub(T, d)
    subs = collect(Iterators.take(Combinatorics.combinations(1:length(B), k), s))
    sub = subs[end]
    return B[sub]
end
mub(d::Integer, k::Integer, s::Integer = 1) = mub(ComplexF64, d, k, s)

"""
    test_mub(B::Vector{Matrix{<:Number}})

Checks if the input bases are mutually unbiased.
"""
function test_mub(B::Vector{Matrix{T}}) where {T<:Number}
    d = size(B[1], 1)
    k = length(B)
    inv_d = inv(T(d))
    for x ∈ 1:k, y ∈ x:k, a ∈ 1:d, b ∈ 1:d
        # expected scalar product squared
        if x == y
            sc2_exp = T(a == b)
        else
            sc2_exp = inv_d
        end
        sc2 = abs2(dot(B[x][:, a], B[y][:, b]))
        if abs2(sc2 - sc2_exp) > _eps(T)
            return false
        end
    end
    return true
end
export test_mub

"""
    povm(B::Vector{<:AbstractMatrix{T}})

Creates a set of (projective) measurements from a set of bases given as unitary matrices.
"""
function povm(B::Vector{<:AbstractMatrix})
    return [[ketbra(B[x][:, a]) for a ∈ 1:size(B[x], 2)] for x ∈ eachindex(B)]
end
export povm

"""
    tensor_to_povm(A::Array{T,4}, o::Vector{Int})

Converts a set of measurements in the common tensor format into a matrix of (hermitian) matrices.
By default, the second argument is fixed by the size of `A`.
It can also contain custom number of outcomes if there are measurements with less outcomes.
"""
function tensor_to_povm(Aax::Array{T,4}, o::Vector{Int} = fill(size(Aax, 3), size(Aax, 4))) where {T}
    return [[Hermitian(Aax[:, :, a, x]) for a ∈ 1:o[x]] for x ∈ axes(Aax, 4)]
end
export tensor_to_povm

"""
    povm_to_tensor(Axa::Vector{<:Measurement})

Converts a matrix of (hermitian) matrices into a set of measurements in the common tensor format.
"""
function povm_to_tensor(Axa::Vector{Measurement{T}}) where {T<:Number}
    d, o, m = _measurements_parameters(Axa)
    Aax = zeros(T, d, d, maximum(o), m)
    for x ∈ eachindex(Axa)
        for a ∈ eachindex(Axa[x])
            Aax[:, :, a, x] .= Axa[x][a]
        end
    end
    return Aax
end
export povm_to_tensor

function _measurements_parameters(Axa::Vector{Measurement{T}}) where {T<:Number}
    @assert !isempty(Axa)
    # dimension on which the measurements act
    d = size(Axa[1][1], 1)
    # tuple of outcome numbers
    o = Tuple(length.(Axa))
    # number of inputs, i.e., of mesurements
    m = length(Axa)
    return d, o, m
end
_measurements_parameters(Aa::Measurement) = _measurements_parameters([Aa])

"""
    test_povm(A::Vector{<:AbstractMatrix{T}})

Checks if the measurement defined by A is valid (hermitian, semi-definite positive, and normalized).
"""
function test_povm(E::Vector{<:AbstractMatrix{T}}) where {T<:Number}
    !all(ishermitian.(E)) && return false
    d = size(E[1], 1)
    !(sum(E) ≈ I(d)) && return false
    for i ∈ 1:length(E)
        minimum(eigvals(E[i])) < -_rtol(T) && return false
    end
    return true
end
export test_povm

"""
    sic_povm([T=ComplexF64,] d::Integer)

Constructs a vector of `d²` vectors |vᵢ⟩ such that |vᵢ⟩⟨vᵢ| forms a SIC-POVM of dimension `d`.
This construction is based on the Weyl-Heisenberg fiducial.

Reference: Appleby, Yadsan-Appleby, Zauner, [arXiv:1209.1813](http://arxiv.org/abs/1209.1813)
"""
function sic_povm(::Type{T}, d::Integer) where {T}
    fiducial = _fiducial_WH(T, d)
    vecs = Vector{Vector{T}}(undef, d^2)
    for p ∈ 0:d-1, q ∈ 0:d-1
        vecs[d*p+q+1] = shiftclock(fiducial, p, q)
    end
    sqrt_d = _sqrt(T, d)
    for vi ∈ vecs
        vi ./= sqrt_d * norm(vi)
    end
    return vecs
end
sic_povm(d::Integer) = sic_povm(ComplexF64, d)
export sic_povm

"""
    test_sic(vecs)

Checks if `vecs` is a vector of `d²` vectors |vᵢ⟩ such that |vᵢ⟩⟨vᵢ| forms a SIC-POVM of dimension `d`.
"""
function test_sic(vecs::Vector{Vector{T}}) where {T<:Number}
    d = length(vecs[1])
    length(vecs) == d^2 || throw(ArgumentError("Number of vectors must be d² = $(d^2), got $(length(vecs))."))
    normalization = inv(T(d^2))
    symmetry = inv(T(d^2 * (d + 1)))
    for j ∈ 1:d^2, i ∈ 1:j
        inner_product = abs2(dot(vecs[i], vecs[j]))
        if i == j
            deviation = abs2(inner_product - normalization)
        else
            deviation = abs2(inner_product - symmetry)
        end
        if deviation > _eps(T)
            return false
        end
    end
    return true
end
export test_sic

"""
    dilate_povm(vecs::Vector{Vector{T}})

Does the Naimark dilation of a rank-1 POVM given as a vector of vectors. This is the minimal dilation.
"""
function dilate_povm(vecs::Vector{Vector{T}}) where {T<:Number}
    d = length(vecs[1])
    n = length(vecs)
    V = Matrix{T}(undef, n, d)
    for j ∈ 1:d
        for i ∈ 1:n
            V[i, j] = conj(vecs[i][j])
        end
    end
    return V
end
export dilate_povm

"""
    dilate_povm(E::Vector{<:AbstractMatrix})

Does the Naimark dilation of a POVM given as a vector of matrices.
This always works, but is wasteful if the POVM elements are not full rank.
"""
function dilate_povm(E::Measurement{T}) where {T}
    n = length(E)
    rtE = sqrt.(E)::Measurement{T} #implicit test of whether E is psd
    return sum(kron(rtE[i], ket(i, n)) for i ∈ 1:n)
end

"""
    _fiducial_WH([T=ComplexF64,] d::Integer)

Computes the fiducial Weyl-Heisenberg vector of dimension `d`.

References:
- Appleby, Yadsan-Appleby, Zauner, [arXiv:1209.1813](http://arxiv.org/abs/1209.1813)
- [www.gerhardzauner.at/sicfiducials.html](http://www.gerhardzauner.at/sicfiducials.html)
"""
function _fiducial_WH(::Type{T}, d::Integer) where {T}
    R = real(T)
    if d == 1
        return T[1]
    elseif d == 2
        return T[sqrt((1 + 1 / sqrt(R(3))) / 2), exp(im * R(π) / 4) * sqrt((1 - 1 / sqrt(R(3))) / 2)]
    elseif d == 3
        return T[0, 1, 1]
    elseif d == 4
        a = sqrt(R(5))
        r = sqrt(R(2))
        b = im * sqrt(a + 1)
        return T[
            R(1) / 40 * (-a + 5) * r + R(1) / 20 * (-a + 5),
            ((-R(1) / 40 * a * r - R(1) / 40 * a) * b + (R(1) / 80 * (a - 5) * r + R(1) / 40 * (a - 5))) * im +
            -R(1) / 40 * a * b +
            R(1) / 80 * (-a + 5) * r,
            R(1) / 40 * (a - 5) * r * im,
            ((R(1) / 40 * a * r + R(1) / 40 * a) * b + (R(1) / 80 * (a - 5) * r + R(1) / 40 * (a - 5))) * im +
            R(1) / 40 * a * b +
            R(1) / 80 * (-a + 5) * r
        ]
    elseif d == 5
        a = sqrt(R(3))
        r = sqrt(R(5))
        t = sin(R(π) / 5)
        b = im * sqrt(5a + (5 + 3r) * t)
        return T[
            R(1) / 60 * (-a + 3) * r + R(1) / 12 * (-a + 3),
            (
                ((R(1) / 120 * a * r + R(1) / 40 * a) * t + (-R(1) / 240 * a * r + R(1) / 240 * a)) * b +
                (R(1) / 120 * (a - 3) * r * t + (R(1) / 32 * (-a + 1) * r + R(1) / 32 * (-a + 1)))
            ) * im +
            ((-R(1) / 120 * a * r - R(1) / 120 * a) * t - R(1) / 120 * a) * b +
            R(1) / 40 * (a - 1) * r * t +
            R(1) / 160 * (-a + 3) * r +
            R(1) / 96 * (a - 3),
            (
                (
                    (-R(1) / 80 * a * r + R(1) / 240 * (-7a + 6)) * t +
                    (R(1) / 480 * (-a + 9) * r + R(1) / 480 * (-a + 15))
                ) * b + (R(1) / 120 * (-a + 3) * r * t + (R(1) / 32 * (-a + 1) * r + R(1) / 32 * (-a + 1)))
            ) * im +
            (
                (R(1) / 240 * (2a + 3) * r + R(1) / 240 * (4a + 3)) * t +
                (R(1) / 480 * (-a - 3) * r + R(1) / 160 * (-a - 5))
            ) * b +
            R(1) / 40 * (-a + 1) * r * t +
            R(1) / 160 * (-a + 3) * r +
            R(1) / 96 * (a - 3),
            (
                ((R(1) / 80 * a * r + R(1) / 240 * (7a - 6)) * t + (R(1) / 480 * (a - 9) * r + R(1) / 480 * (a - 15))) *
                b + (R(1) / 120 * (-a + 3) * r * t + (R(1) / 32 * (-a + 1) * r + R(1) / 32 * (-a + 1)))
            ) * im +
            (
                (R(1) / 240 * (-2a - 3) * r + R(1) / 240 * (-4a - 3)) * t +
                (R(1) / 480 * (a + 3) * r + R(1) / 160 * (a + 5))
            ) * b +
            R(1) / 40 * (-a + 1) * r * t +
            R(1) / 160 * (-a + 3) * r +
            R(1) / 96 * (a - 3),
            (
                ((-R(1) / 120 * a * r - R(1) / 40 * a) * t + (R(1) / 240 * a * r - R(1) / 240 * a)) * b +
                (R(1) / 120 * (a - 3) * r * t + (R(1) / 32 * (-a + 1) * r + R(1) / 32 * (-a + 1)))
            ) * im +
            ((R(1) / 120 * a * r + R(1) / 120 * a) * t + R(1) / 120 * a) * b +
            R(1) / 40 * (a - 1) * r * t +
            R(1) / 160 * (-a + 3) * r +
            R(1) / 96 * (a - 3)
        ]
    elseif d == 6
        a = sqrt(R(21))
        r = [sqrt(R(2)), sqrt(R(3))]
        b = [im * sqrt(2a + 6), 2real((1 + im * sqrt(R(7)))^(R(1) / 3))]
        return T[
            (
                R(1) / 1008 * (-a + 3) * r[2] * b[1] * b[2]^2 +
                R(1) / 504 * (a - 6) * r[2] * b[1] * b[2] +
                (R(1) / 168 * (a - 2) * r[2] - R(1) / 168 * a) * b[1]
            ) * im +
            R(1) / 504 * (-a + 5) * r[2] * b[2]^2 +
            R(1) / 504 * (a + 1) * r[2] * b[2] +
            R(1) / 504 * (-a - 13) * r[2] +
            R(1) / 168 * (a + 21),
            (
                (R(1) / 2016 * (-a + 3) * r[2] * b[1] + (R(1) / 504 * (a - 7) * r[2] + R(1) / 336 * (a - 5))) * b[2]^2 +
                (
                    (R(1) / 1008 * (a - 6) * r[2] + R(1) / 672 * (-a + 7)) * b[1] +
                    (R(1) / 1008 * (-a - 7) * r[2] + R(1) / 336 * (-a - 1))
                ) * b[2] +
                (
                    (R(1) / 672 * (a + 3) * r[2] + R(1) / 672 * (-3a + 7)) * b[1] +
                    (R(1) / 504 * (-3a + 14) * r[2] + R(1) / 168 * (-a - 4))
                )
            ) * im +
            (
                (R(1) / 1008 * (-a + 3) * r[2] + R(1) / 672 * (a - 3)) * b[1] +
                (R(1) / 1008 * (-3a + 7) * r[2] - R(1) / 84)
            ) * b[2]^2 +
            (
                (R(1) / 2016 * (a - 3) * r[2] - R(1) / 336) * b[1] +
                (R(1) / 1008 * (5a - 7) * r[2] + R(1) / 336 * (a - 5))
            ) * b[2] +
            (R(1) / 224 * (a - 5) * r[2] + R(1) / 672 * (-7a + 19)) * b[1] +
            R(1) / 72 * (a - 4) * r[2] +
            R(1) / 168 * (-a + 22),
            (
                (R(1) / 672 * (-a + 3) * r[2] * b[1] + R(1) / 336 * (a - 5)) * b[2]^2 +
                (R(1) / 336 * r[2] * b[1] + R(1) / 336 * (-a - 1)) * b[2] +
                (
                    (R(1) / 672 * (5a - 19) * r[2] + R(1) / 224 * (-a + 7)) * b[1] +
                    (R(1) / 336 * (5a - 21) * r[2] + R(1) / 336 * (-5a + 13))
                )
            ) * im +
            (R(1) / 672 * (-a + 3) * b[1] + R(1) / 1008 * (a - 5) * r[2]) * b[2]^2 +
            (R(1) / 336 * b[1] + R(1) / 1008 * (-a - 1) * r[2]) * b[2] +
            (R(1) / 672 * (-a + 7) * r[2] + R(1) / 672 * (5a - 19)) * b[1] +
            R(1) / 1008 * (-5a + 13) * r[2] +
            R(1) / 336 * (5a - 21),
            (
                (R(1) / 504 * (a - 3) * r[2] * b[1] + R(1) / 504 * (a - 1) * r[2]) * b[2]^2 +
                (R(1) / 1008 * (-a + 3) * r[2] * b[1] + R(1) / 252 * (-a + 2) * r[2]) * b[2] +
                (R(1) / 168 * (-a + 4) * r[2] * b[1] + R(1) / 504 * (-9a + 11) * r[2])
            ) * im +
            (R(1) / 1008 * (a - 3) * r[2] * b[1] - R(1) / 126 * r[2]) * b[2]^2 +
            (R(1) / 1008 * (a - 9) * r[2] * b[1] + R(1) / 504 * (a - 5) * r[2]) * b[2] +
            R(1) / 168 * (-a + 2) * r[2] * b[1] +
            R(1) / 504 * (-5a + 23) * r[2],
            (
                (R(1) / 2016 * (-a + 3) * r[2] * b[1] + R(1) / 336 * (-a + 1)) * b[2]^2 +
                (R(1) / 2016 * (-a + 9) * r[2] * b[1] + R(1) / 168 * (a - 2)) * b[2] +
                (
                    (R(1) / 672 * (a + 3) * r[2] + R(1) / 672 * (a - 21)) * b[1] +
                    (R(1) / 168 * a * r[2] + R(1) / 168 * (3a - 16))
                )
            ) * im +
            (R(1) / 672 * (-a + 3) * b[1] + R(1) / 1008 * (a + 7) * r[2]) * b[2]^2 +
            (R(1) / 672 * (a - 5) * b[1] + R(1) / 504 * (-2a + 7) * r[2]) * b[2] +
            (R(1) / 672 * (3a - 7) * r[2] + R(1) / 672 * (a - 5)) * b[1] +
            R(1) / 504 * (-a - 28) * r[2] +
            R(1) / 168 * a,
            (
                (R(1) / 672 * (a - 3) * b[1] + (R(1) / 1008 * (-a + 1) * r[2] + R(1) / 84)) * b[2]^2 +
                (
                    (R(1) / 672 * (a - 7) * r[2] + R(1) / 672 * (-a + 5)) * b[1] +
                    (R(1) / 504 * (a - 2) * r[2] + R(1) / 336 * (-a + 5))
                ) * b[2] +
                (
                    (R(1) / 672 * (a - 7) * r[2] + R(1) / 672 * (-3a + 5)) * b[1] +
                    (R(1) / 1008 * (3a - 11) * r[2] + R(1) / 336 * (-a - 23))
                )
            ) * im +
            (R(1) / 672 * (-a + 3) * r[2] * b[1] + (R(1) / 252 * r[2] + R(1) / 336 * (a - 1))) * b[2]^2 +
            (
                (R(1) / 672 * (a - 5) * r[2] + R(1) / 672 * (a - 7)) * b[1] +
                (R(1) / 1008 * (-a + 5) * r[2] + R(1) / 168 * (-a + 2))
            ) * b[2] +
            (R(1) / 672 * (3a - 5) * r[2] + R(1) / 672 * (a - 7)) * b[1] +
            R(1) / 1008 * (-a - 23) * r[2] +
            R(1) / 336 * (-3a + 11)
        ]
    elseif d == 7
        a = sqrt(R(2))
        r = sqrt(R(7))
        t = cos(R(π) / 7)
        b = im * sqrt(2a + 1)
        return T[
            R(1) / 14 * (a + 1),
            (
                (R(1) / 196 * (a - 4) * r * t^2 + R(1) / 392 * (3a + 2) * r * t + R(1) / 392 * (a + 3) * r) * b +
                (R(1) / 196 * (a + 6) * r * t^2 + R(1) / 392 * (-3a - 4) * r * t + R(1) / 392 * (-5a - 9) * r)
            ) * im +
            (-R(1) / 28 * a * t^2 + R(1) / 56 * (a - 2) * t + R(1) / 56 * (a - 1)) * b +
            R(1) / 28 * (3a + 6) * t^2 +
            R(1) / 56 * (-a - 4) * t +
            R(1) / 56 * (-3a - 5),
            (
                (R(1) / 196 * (3a + 2) * r * t^2 + R(1) / 196 * (-2a + 1) * r * t + R(1) / 784 * (a - 4) * r) * b +
                (R(1) / 196 * (-3a - 4) * r * t^2 + R(1) / 196 * (a - 1) * r * t + R(1) / 784 * (-5a - 2) * r)
            ) * im +
            (R(1) / 28 * (a - 2) * t^2 + R(1) / 28 * t - R(1) / 112 * a) * b +
            R(1) / 28 * (-a - 4) * t^2 +
            R(1) / 28 * (-a - 1) * t +
            R(1) / 112 * (a + 6),
            (
                (R(1) / 98 * (2a - 1) * r * t^2 + R(1) / 392 * (-a + 4) * r * t + R(1) / 784 * (-11a + 2) * r) * b +
                (R(1) / 98 * (a - 1) * r * t^2 + R(1) / 392 * (a + 6) * r * t + R(1) / 784 * (-13a - 8) * r)
            ) * im +
            (-R(1) / 14 * t^2 + R(1) / 56 * a * t + R(1) / 112 * (-a + 6)) * b +
            R(1) / 14 * (-a - 1) * t^2 +
            R(1) / 56 * (3a + 6) * t +
            R(1) / 112 * a,
            (
                (R(1) / 98 * (-2a + 1) * r * t^2 + R(1) / 392 * (a - 4) * r * t + R(1) / 784 * (11a - 2) * r) * b +
                (R(1) / 98 * (a - 1) * r * t^2 + R(1) / 392 * (a + 6) * r * t + R(1) / 784 * (-13a - 8) * r)
            ) * im +
            (R(1) / 14 * t^2 - R(1) / 56 * a * t + R(1) / 112 * (a - 6)) * b +
            R(1) / 14 * (-a - 1) * t^2 +
            R(1) / 56 * (3a + 6) * t +
            R(1) / 112 * a,
            (
                (R(1) / 196 * (-3a - 2) * r * t^2 + R(1) / 196 * (2a - 1) * r * t + R(1) / 784 * (-a + 4) * r) * b +
                (R(1) / 196 * (-3a - 4) * r * t^2 + R(1) / 196 * (a - 1) * r * t + R(1) / 784 * (-5a - 2) * r)
            ) * im +
            (R(1) / 28 * (-a + 2) * t^2 - R(1) / 28 * t + R(1) / 112 * a) * b +
            R(1) / 28 * (-a - 4) * t^2 +
            R(1) / 28 * (-a - 1) * t +
            R(1) / 112 * (a + 6),
            (
                (R(1) / 196 * (-a + 4) * r * t^2 + R(1) / 392 * (-3a - 2) * r * t + R(1) / 392 * (-a - 3) * r) * b +
                (R(1) / 196 * (a + 6) * r * t^2 + R(1) / 392 * (-3a - 4) * r * t + R(1) / 392 * (-5a - 9) * r)
            ) * im +
            (R(1) / 28 * a * t^2 + R(1) / 56 * (-a + 2) * t + R(1) / 56 * (-a + 1)) * b +
            R(1) / 28 * (3a + 6) * t^2 +
            R(1) / 56 * (-a - 4) * t +
            R(1) / 56 * (-3a - 5)
        ]
    elseif d == 8
        a = sqrt(R(5))
        r1 = sqrt(R(2))
        t = cos(R(π) / 8)
        b1 = im * sqrt(a - 1)
        return T[
            R(1) / 24 * (-a + 3),
            ((R(1) / 24 * t - R(1) / 48 * r1) * b1 + (R(1) / 48 * (-a + 3) * r1 * t + R(1) / 48 * (-a + 3))) * im +
            ((R(1) / 24 * r1 - R(1) / 24) * t - R(1) / 48 * r1) * b1 +
            (R(1) / 48 * (a - 3) * r1 + R(1) / 24 * (-a + 3)) * t +
            R(1) / 48 * (a - 3),
            -R(1) / 24 * r1 * b1 * im,
            ((R(1) / 24 * t + R(1) / 48 * r1) * b1 + (R(1) / 48 * (a - 3) * r1 * t + R(1) / 48 * (-a + 3))) * im +
            ((R(1) / 24 * r1 - R(1) / 24) * t + R(1) / 48 * r1) * b1 +
            (R(1) / 48 * (-a + 3) * r1 + R(1) / 24 * (a - 3)) * t +
            R(1) / 48 * (a - 3),
            R(1) / 24 * (a - 3),
            ((-R(1) / 24 * t - R(1) / 48 * r1) * b1 + (R(1) / 48 * (a - 3) * r1 * t + R(1) / 48 * (-a + 3))) * im +
            ((-R(1) / 24 * r1 + R(1) / 24) * t - R(1) / 48 * r1) * b1 +
            (R(1) / 48 * (-a + 3) * r1 + R(1) / 24 * (a - 3)) * t +
            R(1) / 48 * (a - 3),
            R(1) / 24 * r1 * b1 * im,
            ((-R(1) / 24 * t + R(1) / 48 * r1) * b1 + (R(1) / 48 * (-a + 3) * r1 * t + R(1) / 48 * (-a + 3))) * im +
            ((-R(1) / 24 * r1 + R(1) / 24) * t + R(1) / 48 * r1) * b1 +
            (R(1) / 48 * (a - 3) * r1 + R(1) / 24 * (-a + 3)) * t +
            R(1) / 48 * (a - 3)
        ]
    elseif d == 9
        a = sqrt(R(15))
        r1 = sqrt(R(3))
        t = cos(R(π) / 9)
        b1 = im * sqrt(2 * a + 4 * r1)
        b2 = 2 * real((6 + 6 * im * sqrt(R(5)))^(R(1) / 3))
        return T[
            (R(1) / 72 * r1 - R(1) / 120 * a) * b1 * im + R(1) / 360 * (a - 5) * r1 + R(1) / 120 * (a + 5),
            (
                (
                    (
                        (R(1) / 1620 * (a - 3) * r1 + R(1) / 540 * (a - 3)) * t^2 +
                        (R(1) / 3240 * (a + 3) * r1 + R(1) / 2160 * (-a - 3)) * t +
                        (R(1) / 3240 * (-a + 3) * r1 + R(1) / 1080 * (-a + 3))
                    ) * b1 + (R(1) / 90 * t^2 - R(1) / 360 * t - R(1) / 180)
                ) * b2^2 +
                (
                    (
                        (R(1) / 90 * r1 + R(1) / 180 * (-a + 1)) * t^2 +
                        (-R(1) / 180 * r1 + R(1) / 360 * (a - 2)) * t +
                        (-R(1) / 180 * r1 + R(1) / 360 * (a - 1))
                    ) * b1 + (-R(1) / 90 * t^2 + R(1) / 360 * (-a + 1) * t + R(1) / 180)
                ) * b2 +
                (
                    (
                        (R(1) / 270 * (-a + 1) * r1 + R(1) / 45 * (-a + 3)) * t^2 +
                        (R(1) / 1080 * (-4 * a - 7) * r1 + R(1) / 360 * (a + 6)) * t +
                        (R(1) / 1080 * (3 * a - 17) * r1 + R(1) / 60 * (a - 2))
                    ) * b1 + (
                        (R(1) / 90 * (-a + 5) * r1 + R(1) / 90 * (a - 17)) * t^2 +
                        (R(1) / 360 * (-a - 5) * r1 + R(1) / 360 * (-a + 17)) * t +
                        (-R(1) / 36 * r1 + R(1) / 180 * (-a + 17))
                    )
                )
            ) * im +
            (
                (
                    (-R(1) / 270 * r1 + R(1) / 540 * (a + 1)) * t^2 +
                    (R(1) / 2160 * (-a + 5) * r1 + R(1) / 1080 * (-a + 1)) * t +
                    (R(1) / 540 * r1 + R(1) / 1080 * (-a - 1))
                ) * b1 - R(1) / 360 * r1 * t
            ) * b2^2 +
            (
                (
                    (R(1) / 540 * (a - 3) * r1 - R(1) / 90) * t^2 +
                    (R(1) / 1080 * a * r1 - R(1) / 180) * t +
                    (R(1) / 1080 * (-a + 3) * r1 + R(1) / 180)
                ) * b1 + (-R(1) / 270 * a * r1 * t^2 + R(1) / 1080 * (a + 3) * r1 * t + R(1) / 540 * a * r1)
            ) * b2 +
            (
                (R(1) / 270 * (a + 12) * r1 + R(1) / 270 * (-7 * a - 6)) * t^2 +
                (R(1) / 216 * (a - 6) * r1 + R(1) / 1080 * (16 * a - 27)) * t +
                (R(1) / 1080 * (-5 * a - 24) * r1 + R(1) / 1080 * (11 * a + 42))
            ) * b1 +
            R(1) / 45 * a * t^2 +
            (R(1) / 360 * (a + 7) * r1 + R(1) / 72 * (-a + 3)) * t +
            R(1) / 360 * (-a + 5) * r1 +
            R(1) / 360 * (-a - 15),
            (
                (
                    (
                        (R(1) / 6480 * (-a + 15) * r1 + R(1) / 2160 * (-a - 3)) * t^2 +
                        (R(1) / 12960 * (-7 * a + 9) * r1 + R(1) / 4320 * (-a + 9)) * t +
                        (R(1) / 12960 * (a - 15) * r1 + R(1) / 4320 * (a + 3))
                    ) * b1 + (
                        (R(1) / 1080 * (a - 6) * r1 + R(1) / 360) * t^2 +
                        (R(1) / 2160 * a * r1 + R(1) / 720) * t +
                        (R(1) / 2160 * (-a + 6) * r1 - R(1) / 720)
                    )
                ) * b2^2 +
                (
                    (
                        (R(1) / 1080 * (2 * a - 15) * r1 + R(1) / 360 * (a - 2)) * t^2 +
                        (R(1) / 2160 * (-a + 6) * r1 + R(1) / 720) * t +
                        (R(1) / 2160 * (-2 * a + 15) * r1 + R(1) / 720 * (-a + 2))
                    ) * b1 + (
                        (R(1) / 360 * (-a + 7) * r1 + R(1) / 360 * (a - 1)) * t^2 +
                        (R(1) / 720 * (a - 5) * r1 + R(1) / 720 * (-a - 1)) * t +
                        (R(1) / 720 * (a - 7) * r1 + R(1) / 720 * (-a + 1))
                    )
                ) * b2 +
                (
                    (
                        (R(1) / 540 * (a - 20) * r1 + R(1) / 180 * (2 * a + 3)) * t^2 +
                        (R(1) / 1080 * (9 * a - 29) * r1 + R(1) / 120 * (a - 3)) * t +
                        (R(1) / 2160 * (-4 * a + 55) * r1 + R(1) / 720 * (-5 * a - 6))
                    ) * b1 + (
                        (R(1) / 180 * (-3 * a + 7) * r1 + R(1) / 180 * (-a - 1)) * t^2 +
                        (R(1) / 180 * (-2 * a + 5) * r1 + R(1) / 180 * (a - 8)) * t +
                        (R(1) / 720 * (7 * a - 29) * r1 + R(1) / 720 * (-a + 17))
                    )
                )
            ) * im +
            (
                (
                    (R(1) / 2160 * (-a + 5) * r1 + R(1) / 2160 * (-5 * a + 11)) * t^2 +
                    (R(1) / 4320 * (a - 1) * r1 + R(1) / 4320 * (3 * a - 13)) * t +
                    (R(1) / 4320 * (a - 5) * r1 + R(1) / 4320 * (5 * a - 11))
                ) * b1 + (
                    (R(1) / 360 * r1 + R(1) / 360 * (a - 2)) * t^2 +
                    (-R(1) / 720 * r1 + R(1) / 720 * (-a + 4)) * t +
                    (-R(1) / 720 * r1 + R(1) / 720 * (-a + 2))
                )
            ) * b2^2 +
            (
                (
                    (R(1) / 1080 * a * r1 - R(1) / 360) * t^2 +
                    (R(1) / 2160 * (-2 * a + 3) * r1 + R(1) / 720 * (-a + 8)) * t +
                    (-R(1) / 2160 * a * r1 + R(1) / 720)
                ) * b1 + (
                    (R(1) / 1080 * (-a - 3) * r1 + R(1) / 360 * (a - 3)) * t^2 +
                    (R(1) / 2160 * (-a + 3) * r1 + R(1) / 720 * (a - 9)) * t +
                    (R(1) / 2160 * (a + 3) * r1 + R(1) / 720 * (-a + 3))
                )
            ) * b2 +
            (
                (R(1) / 540 * (-2 * a - 15) * r1 + R(1) / 540 * (11 * a + 12)) * t^2 +
                (R(1) / 1080 * (a + 3) * r1 + R(1) / 1080 * (-7 * a + 9)) * t +
                (R(1) / 2160 * (a + 30) * r1 + R(1) / 2160 * (-16 * a - 39))
            ) * b1 +
            (R(1) / 180 * (-a - 1) * r1 + R(1) / 180 * (-a - 3)) * t^2 +
            (R(1) / 60 * r1 + R(1) / 180 * (a - 12)) * t +
            R(1) / 240 * (a - 1) * r1 +
            R(1) / 720 * (-7 * a + 21),
            (R(1) / 360 * a * r1 + R(1) / 360 * (2 * a - 15)) * b1 +
            R(1) / 360 * (a - 5) * r1 +
            R(1) / 120 * (-3 * a + 5),
            (
                (
                    (
                        (-R(1) / 810 * a * r1 + R(1) / 1080 * (-a + 9)) * t^2 +
                        (R(1) / 3240 * (a - 3) * r1 + R(1) / 1080 * (a - 3)) * t +
                        (R(1) / 1620 * a * r1 + R(1) / 2160 * (a - 9))
                    ) * b1 + (-R(1) / 180 * t^2 + R(1) / 180 * t + R(1) / 360)
                ) * b2^2 +
                (
                    (R(1) / 180 * t^2 + (R(1) / 180 * r1 + R(1) / 360 * (-a + 1)) * t - R(1) / 360) * b1 +
                    (R(1) / 180 * (a + 1) * t^2 - R(1) / 180 * t + R(1) / 360 * (-a - 1))
                ) * b2 +
                (
                    (
                        (R(1) / 540 * (6 * a + 5) * r1 + R(1) / 60 * (a - 6)) * t^2 +
                        (R(1) / 540 * (-a + 1) * r1 + R(1) / 90 * (-a + 3)) * t +
                        (R(1) / 216 * (-a - 4) * r1 + R(1) / 360 * (-a + 18))
                    ) * b1 + (
                        (R(1) / 180 * (3 * a - 5) * r1 + R(1) / 180 * (-a + 17)) * t^2 +
                        (R(1) / 180 * (-a + 5) * r1 + R(1) / 180 * (a - 17)) * t +
                        (R(1) / 72 * (-a + 1) * r1 + R(1) / 360 * (a - 17))
                    )
                )
            ) * im +
            (
                (
                    (R(1) / 1080 * (a - 1) * r1 - R(1) / 270) * t^2 +
                    (-R(1) / 540 * r1 + R(1) / 1080 * (a + 1)) * t +
                    (R(1) / 2160 * (-a + 1) * r1 + R(1) / 540)
                ) * b1 + (R(1) / 180 * r1 * t^2 - R(1) / 360 * r1)
            ) * b2^2 +
            (
                (
                    (R(1) / 540 * (-2 * a + 3) * r1 + R(1) / 45) * t^2 +
                    (R(1) / 1080 * (a - 3) * r1 - R(1) / 180) * t +
                    (R(1) / 1080 * (2 * a - 3) * r1 - R(1) / 90)
                ) * b1 + (R(1) / 540 * (a - 3) * r1 * t^2 - R(1) / 540 * a * r1 * t + R(1) / 1080 * (-a + 3) * r1)
            ) * b2 +
            (
                (R(1) / 540 * (-7 * a + 6) * r1 + R(1) / 540 * (-2 * a + 39)) * t^2 +
                (R(1) / 540 * (a + 12) * r1 + R(1) / 540 * (-7 * a - 6)) * t +
                (R(1) / 540 * (2 * a - 3) * r1 + R(1) / 1080 * (-a - 9))
            ) * b1 +
            (R(1) / 180 * (-a - 7) * r1 + R(1) / 180 * (a - 15)) * t^2 +
            R(1) / 90 * a * t +
            R(1) / 30 * r1 +
            R(1) / 180 * a,
            (
                (
                    (
                        (R(1) / 6480 * (-7 * a + 9) * r1 + R(1) / 2160 * (-a + 9)) * t^2 +
                        (R(1) / 1620 * (a - 3) * r1 + R(1) / 2160 * (a - 3)) * t +
                        (R(1) / 12960 * (7 * a - 9) * r1 + R(1) / 4320 * (a - 9))
                    ) * b1 + (
                        (R(1) / 1080 * a * r1 + R(1) / 360) * t^2 +
                        (R(1) / 1080 * (-a + 3) * r1 - R(1) / 360) * t +
                        (-R(1) / 2160 * a * r1 - R(1) / 720)
                    )
                ) * b2^2 +
                (
                    (
                        (R(1) / 1080 * (-a + 6) * r1 + R(1) / 360) * t^2 +
                        (R(1) / 2160 * (-a + 9) * r1 + R(1) / 720 * (-a + 1)) * t +
                        (R(1) / 2160 * (a - 6) * r1 - R(1) / 720)
                    ) * b1 + (
                        (R(1) / 360 * (a - 5) * r1 + R(1) / 360 * (-a - 1)) * t^2 +
                        (-R(1) / 360 * r1 + R(1) / 360) * t +
                        (R(1) / 720 * (-a + 5) * r1 + R(1) / 720 * (a + 1))
                    )
                ) * b2 +
                (
                    (
                        (R(1) / 540 * (9 * a - 29) * r1 + R(1) / 60 * (a - 3)) * t^2 +
                        (R(1) / 1080 * (-10 * a + 49) * r1 + R(1) / 360 * (-5 * a + 6)) * t +
                        (R(1) / 2160 * (-20 * a + 73) * r1 + R(1) / 720 * (-7 * a + 18))
                    ) * b1 + (
                        (R(1) / 90 * (-2 * a + 5) * r1 + R(1) / 90 * (a - 8)) * t^2 +
                        (R(1) / 360 * (7 * a - 17) * r1 + R(1) / 360 * (-a + 17)) * t +
                        (R(1) / 720 * (9 * a - 35) * r1 + R(1) / 720 * (-7 * a + 47))
                    )
                )
            ) * im +
            (
                (
                    (R(1) / 2160 * (a - 1) * r1 + R(1) / 2160 * (3 * a - 13)) * t^2 +
                    (-R(1) / 1080 * r1 + R(1) / 2160 * (a + 1)) * t +
                    (R(1) / 4320 * (-a + 1) * r1 + R(1) / 4320 * (-3 * a + 13))
                ) * b1 + (
                    (-R(1) / 360 * r1 + R(1) / 360 * (-a + 4)) * t^2 - R(1) / 360 * t +
                    R(1) / 720 * r1 +
                    R(1) / 720 * (a - 4)
                )
            ) * b2^2 +
            (
                (
                    (R(1) / 1080 * (-2 * a + 3) * r1 + R(1) / 360 * (-a + 8)) * t^2 +
                    (R(1) / 2160 * (a - 3) * r1 + R(1) / 720 * (a - 7)) * t +
                    (R(1) / 2160 * (2 * a - 3) * r1 + R(1) / 720 * (a - 8))
                ) * b1 + (
                    (R(1) / 1080 * (-a + 3) * r1 + R(1) / 360 * (a - 9)) * t^2 +
                    (R(1) / 1080 * a * r1 + R(1) / 360 * (-a + 6)) * t +
                    (R(1) / 2160 * (a - 3) * r1 + R(1) / 720 * (-a + 9))
                )
            ) * b2 +
            (
                (R(1) / 540 * (a + 3) * r1 + R(1) / 540 * (-7 * a + 9)) * t^2 +
                (R(1) / 1080 * (a + 12) * r1 + R(1) / 1080 * (-4 * a - 21)) * t +
                (R(1) / 2160 * (-5 * a - 6) * r1 + R(1) / 2160 * (20 * a - 33))
            ) * b1 +
            (R(1) / 30 * r1 + R(1) / 90 * (a - 12)) * t^2 +
            (R(1) / 360 * (a - 5) * r1 + R(1) / 360 * (-a + 27)) * t +
            R(1) / 720 * (a - 17) * r1 +
            R(1) / 720 * (-13 * a + 63),
            (-R(1) / 72 * r1 + R(1) / 120 * a) * b1 * im +
            (-R(1) / 360 * a * r1 + R(1) / 360 * (-2 * a + 15)) * b1 +
            R(1) / 180 * (-a + 5) * r1 +
            R(1) / 60 * (a - 5),
            (
                (
                    (
                        (R(1) / 1620 * (a + 3) * r1 + R(1) / 1080 * (-a - 3)) * t^2 +
                        (-R(1) / 1620 * a * r1 + R(1) / 2160 * (-a + 9)) * t +
                        (R(1) / 3240 * (-a - 3) * r1 + R(1) / 2160 * (a + 3))
                    ) * b1 + (-R(1) / 180 * t^2 - R(1) / 360 * t + R(1) / 360)
                ) * b2^2 +
                (
                    (
                        (-R(1) / 90 * r1 + R(1) / 180 * (a - 2)) * t^2 +
                        R(1) / 360 * t +
                        (R(1) / 180 * r1 + R(1) / 360 * (-a + 2))
                    ) * b1 + (R(1) / 180 * (-a + 1) * t^2 + R(1) / 360 * (a + 1) * t + R(1) / 360 * (a - 1))
                ) * b2 +
                (
                    (
                        (R(1) / 540 * (-4 * a - 7) * r1 + R(1) / 180 * (a + 6)) * t^2 +
                        (R(1) / 1080 * (6 * a + 5) * r1 + R(1) / 120 * (a - 6)) * t +
                        (R(1) / 1080 * (5 * a - 8) * r1 + R(1) / 360 * (a - 6))
                    ) * b1 + (
                        (R(1) / 180 * (-a - 5) * r1 + R(1) / 180 * (-a + 17)) * t^2 +
                        (R(1) / 360 * (3 * a - 5) * r1 + R(1) / 360 * (-a + 17)) * t +
                        (R(1) / 360 * (-a + 5) * r1 + R(1) / 360 * (a - 17))
                    )
                )
            ) * im +
            (
                (
                    (R(1) / 1080 * (-a + 5) * r1 + R(1) / 540 * (-a + 1)) * t^2 +
                    (R(1) / 2160 * (a - 1) * r1 - R(1) / 540) * t +
                    (R(1) / 2160 * (a - 5) * r1 + R(1) / 1080 * (a - 1))
                ) * b1 + (-R(1) / 180 * r1 * t^2 + R(1) / 360 * r1 * t + R(1) / 360 * r1)
            ) * b2^2 +
            (
                (
                    (R(1) / 540 * a * r1 - R(1) / 90) * t^2 +
                    (R(1) / 1080 * (-2 * a + 3) * r1 + R(1) / 90) * t +
                    (-R(1) / 1080 * a * r1 + R(1) / 180)
                ) * b1 +
                (R(1) / 540 * (a + 3) * r1 * t^2 + R(1) / 1080 * (a - 3) * r1 * t + R(1) / 1080 * (-a - 3) * r1)
            ) * b2 +
            (
                (R(1) / 108 * (a - 6) * r1 + R(1) / 540 * (16 * a - 27)) * t^2 +
                (R(1) / 1080 * (-7 * a + 6) * r1 + R(1) / 1080 * (-2 * a + 39)) * t +
                (R(1) / 540 * (-4 * a + 15) * r1 + R(1) / 1080 * (-19 * a + 57))
            ) * b1 +
            (R(1) / 180 * (a + 7) * r1 + R(1) / 36 * (-a + 3)) * t^2 +
            (R(1) / 360 * (-a - 7) * r1 + R(1) / 360 * (a - 15)) * t +
            R(1) / 180 * (-a - 1) * r1 +
            R(1) / 180 * (4 * a - 15),
            (
                (
                    (
                        (R(1) / 810 * (a - 3) * r1 + R(1) / 1080 * (a - 3)) * t^2 +
                        (R(1) / 12960 * (-a + 15) * r1 + R(1) / 4320 * (-a - 3)) * t +
                        (R(1) / 1620 * (-a + 3) * r1 + R(1) / 2160 * (-a + 3))
                    ) * b1 + (
                        (R(1) / 540 * (-a + 3) * r1 - R(1) / 180) * t^2 +
                        (R(1) / 2160 * (a - 6) * r1 + R(1) / 720) * t +
                        (R(1) / 1080 * (a - 3) * r1 + R(1) / 360)
                    )
                ) * b2^2 +
                (
                    (
                        (R(1) / 1080 * (-a + 9) * r1 + R(1) / 360 * (-a + 1)) * t^2 +
                        (R(1) / 2160 * (2 * a - 15) * r1 + R(1) / 720 * (a - 2)) * t +
                        (R(1) / 2160 * (a - 9) * r1 + R(1) / 720 * (a - 1))
                    ) * b1 + (
                        (-R(1) / 180 * r1 + R(1) / 180) * t^2 +
                        (R(1) / 720 * (-a + 7) * r1 + R(1) / 720 * (a - 1)) * t +
                        (R(1) / 360 * r1 - R(1) / 360)
                    )
                ) * b2 +
                (
                    (
                        (R(1) / 540 * (-10 * a + 49) * r1 + R(1) / 180 * (-5 * a + 6)) * t^2 +
                        (R(1) / 1080 * (a - 20) * r1 + R(1) / 360 * (2 * a + 3)) * t +
                        (R(1) / 2160 * (18 * a - 83) * r1 + R(1) / 240 * (3 * a - 4))
                    ) * b1 + (
                        (R(1) / 180 * (7 * a - 17) * r1 + R(1) / 180 * (-a + 17)) * t^2 +
                        (R(1) / 360 * (-3 * a + 7) * r1 + R(1) / 360 * (-a - 1)) * t +
                        (R(1) / 720 * (-13 * a + 19) * r1 + R(1) / 720 * (-a - 19))
                    )
                )
            ) * im +
            (
                (
                    (-R(1) / 540 * r1 + R(1) / 1080 * (a + 1)) * t^2 +
                    (R(1) / 4320 * (-a + 5) * r1 + R(1) / 4320 * (-5 * a + 11)) * t +
                    (R(1) / 1080 * r1 + R(1) / 2160 * (-a - 1))
                ) * b1 + (-R(1) / 180 * t^2 + (R(1) / 720 * r1 + R(1) / 720 * (a - 2)) * t + R(1) / 360)
            ) * b2^2 +
            (
                (
                    (R(1) / 1080 * (a - 3) * r1 + R(1) / 360 * (a - 7)) * t^2 +
                    (R(1) / 2160 * a * r1 - R(1) / 720) * t +
                    (R(1) / 2160 * (-a + 3) * r1 + R(1) / 720 * (-a + 7))
                ) * b1 + (
                    (R(1) / 540 * a * r1 + R(1) / 180 * (-a + 6)) * t^2 +
                    (R(1) / 2160 * (-a - 3) * r1 + R(1) / 720 * (a - 3)) * t +
                    (-R(1) / 1080 * a * r1 + R(1) / 360 * (a - 6))
                )
            ) * b2 +
            (
                (R(1) / 540 * (a + 12) * r1 + R(1) / 540 * (-4 * a - 21)) * t^2 +
                (R(1) / 1080 * (-2 * a - 15) * r1 + R(1) / 1080 * (11 * a + 12)) * t +
                (R(1) / 2160 * (-5 * a - 24) * r1 + R(1) / 2160 * (14 * a + 27))
            ) * b1 +
            (R(1) / 180 * (a - 5) * r1 + R(1) / 180 * (-a + 27)) * t^2 +
            (R(1) / 360 * (-a - 1) * r1 + R(1) / 360 * (-a - 3)) * t +
            R(1) / 720 * (-a + 5) * r1 +
            R(1) / 720 * (-7 * a - 39)
        ]
    else
        throw(ArgumentError(string("Invalid input dimension d = ", d)))
    end
end

_fiducial_WH(d::Integer) = _fiducial_WH(ComplexF64, d)

"""
    state_discrimination_min_error(
        ρ::Vector{<:AbstractMatrix{T}},
        q::Vector{<:AbstractFloat} = fill(1/length(ρ), length(ρ));
        verbose = false,
        dualize = false,
        solver = Hypatia.Optimizer{_solver_type(T)}
    ) where {T, S<:AbstractFloat}

Return a POVM ``{E_i}_{=1}^N`` and probability P such that if we observe ``E_i`` with average probability P we are in state ``ρ_i``

# Arguments
- ρ: list of state to be discriminated
- q: (optional) list of probabilities associated with the state ρ, if not provided, uniform probability is assumed
"""
function state_discrimination_min_error(
    ρ::Vector{<:AbstractMatrix{T}},
    q::Vector{S} = fill(1/length(ρ),length(ρ));
    verbose = false,
    dualize = false,
    solver = Hypatia.Optimizer{_solver_type(T)}) where {T,S<:AbstractFloat}

    model = JuMP.GenericModel{_solver_type(T)}()    
    N = length(ρ)
    d = maximum(size(ρ[1]))

    @assert length(q) == length(ρ) 
    @assert isapprox(sum(q), 1.0; atol=eps(S))

    E = [JuMP.@variable(model, [1:d, 1:d] in JuMP.HermitianPSDCone()) for i in 1:N]
    JuMP.@constraint(model, E[N] == I - sum(E[1:N-1]))

    JuMP.@objective(
        model,
        Max,
        sum(q[i] * real(dot(ρ[i]',E[i])) for i in 1:N),
    )

    if dualize
        JuMP.set_optimizer(model, Dualization.dual_optimizer(solver; coefficient_type = _solver_type(T)))
    else
        JuMP.set_optimizer(model, solver)
    end
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)

    JuMP.is_solved_and_feasible(model) || throw(error(JuMP.raw_status(model)))
    return [JuMP.value.(e) for e in E], JuMP.objective_value(model)
end
export state_discrimination_min_error

"""
    pretty_good_povm(ρ::Vector{<:AbstractMatrix{T}}) where {T}

Return a POVM `{E_i}_{i=1}^N` using the pretty good measurements method to discriminate between the states `ρ`

Reference: (https://www.math.uwaterloo.ca/~wcleung/co781-presentation-PGM-Lavoie.pdf)
"""
function pretty_good_povm(ρ::Vector{<:AbstractMatrix{T}}) where {T}
    n = length(ρ)
    d = maximum(size(ρ[1]))

    M = sum(ρ)
    rootinvM = Hermitian(M)^-0.5
    E = [Matrix{T}(undef, d, d) for _ ∈ 1:n]
    temp = Matrix{T}(undef, d, d)
    for i ∈ 1:n
        mul!(temp, rootinvM, ρ[i])
        mul!(E[i], temp, rootinvM')
    end
    return Hermitian.(E)
end
export pretty_good_povm