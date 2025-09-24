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
        !isposdef(E[i] + _rtol(T)*I) && return false
    end
    return true
end
export test_povm

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
    discrimination_min_error(
        ρ::Vector{<:AbstractMatrix},
        q::Vector{<:Real} = fill(1/length(ρ), length(ρ));
        verbose = false,
        dualize = false,
        solver = Hypatia.Optimizer
    )

Computes the minimum-error probability of discriminating a vector of states `ρ` with probabilities `q`, along with the optimal POVM. `q` is assumed uniform if omitted.
"""
function discrimination_min_error(
    ρ::Vector{<:AbstractMatrix{T}},
    q::Vector{<:Real} = fill(real(T)(1) / length(ρ), length(ρ));
    verbose = false,
    dualize = false,
    solver = Hypatia.Optimizer{_solver_type(T)}
) where {T}
    is_complex = T <: Complex
    psd_cone, wrapper, hermitian_space = _sdp_parameters(is_complex)
    model = JuMP.GenericModel{_solver_type(T)}()
    N = length(ρ)
    d = size(ρ[1], 1)

    @assert length(q) == length(ρ)

    E = [1 * JuMP.@variable(model, [1:d, 1:d] in psd_cone) for i ∈ 1:N-1]
    E_N = wrapper(I - sum(E))
    JuMP.@constraint(model, E_N in psd_cone)
    push!(E, E_N)

    JuMP.@objective(model, Max, sum(q[i] * real(dot(ρ[i], E[i])) for i ∈ 1:N))

    if dualize
        JuMP.set_optimizer(model, Dualization.dual_optimizer(solver; coefficient_type = _solver_type(T)))
    else
        JuMP.set_optimizer(model, solver)
    end
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)

    JuMP.is_solved_and_feasible(model) || @warn JuMP.raw_status(model)
    return JuMP.objective_value(model), JuMP.value.(E)
end
export discrimination_min_error

"""
    pretty_good_measurement(ρ::Vector{<:AbstractMatrix}, q::Vector{<:Real} = ones(length(ρ)))

Computes the pretty good measurement POVM for discriminating a vector of states `ρ` with probabilities `q`. If `q` is omitted it is assumed uniform.

Reference: Watrous, [Theory of Quantum Information Cp. 3](https://cs.uwaterloo.ca/~watrous/TQI/TQI.3.pdf)
"""
function pretty_good_measurement(ρ::Vector{<:AbstractMatrix{T}}, q::Vector{<:Real} = ones(length(ρ))) where {T}
    n = length(ρ)
    d = size(ρ[1], 1)

    for i ∈ 1:n
        parent(ρ[i]) .*= q[i]
    end
    M = sum(ρ)
    λ, temp = eigen(Hermitian(M))
    map!(x -> x > _rtol(T) ? x^-0.25 : zero(x), λ, λ)
    @inbounds for j ∈ 1:d, i ∈ 1:d
        temp[i, j] *= λ[j]
    end
    rootinvM = temp * temp'
    E = [Matrix{T}(undef, d, d) for _ ∈ 1:n]
    for i ∈ 1:n
        mul!(temp, rootinvM, ρ[i])
        mul!(E[i], temp, rootinvM')
    end
    kernM = I - sum(E)
    E[n] .+= kernM
    return Hermitian.(E)
end
export pretty_good_measurement
