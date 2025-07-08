function _wrapper_applymap(M::AbstractMatrix{T}, ::Type{mapT}) where {T,mapT}
    if isa(M, Hermitian)
        return Hermitian
    elseif isa(M, Symmetric) && JuMP.value_type(T) <: Real && JuMP.value_type(mapT) <: Real
        return Symmetric
    elseif isa(M, Symmetric) && JuMP.value_type(T) <: Real
        return Hermitian
    else
        return identity
    end
end

"""
    applymap(K::Vector{<:AbstractMatrix}, M::AbstractMatrix)

Applies the CP map given by the Kraus operators `K` to the matrix `M`. Preserves sparsity.
"""
function applymap(K::Vector{<:AbstractMatrix{T}}, M::AbstractMatrix{S}) where {T,S}
    dout, din = size(K[1])
    TS = Base.promote_op(*, T, S)
    if all(SA.issparse.(K)) && SA.issparse(M)
        temp = SA.spzeros(TS, dout, din)
        result = SA.spzeros(TS, dout, dout)
    else
        temp = Matrix{TS}(undef, dout, din)
        result = Matrix{TS}(undef, dout, dout)
    end
    applymap!(result, K, M, temp)
    return _wrapper_applymap(M, T)(result)
end
export applymap

"""
    applymap!(result::AbstractMatrix, K::Vector{<:AbstractMatrix}, M::AbstractMatrix, temp::AbstractMatrix)

Applies the CP map given by the Kraus operators `K` to the matrix `M` without allocating or wrapping. `result` and `temp` must be
matrices of size `dout × dout` and `dout × din`, where `dout, din == size(K[1])`.
"""
function applymap!(result::AbstractMatrix, K::Vector{<:AbstractMatrix}, M::AbstractMatrix, temp::AbstractMatrix)
    mul!(temp, K[1], M)
    mul!(result, temp, K[1]')
    for i ∈ 2:length(K)
        mul!(temp, K[i], M)
        mul!(result, temp, K[i]', true, true)
    end
    return result
end
export applymap!

"""
    applymap(Φ::AbstractMatrix, M::AbstractMatrix)

Applies the CP map given by the Choi-Jamiołkowski operator `Φ` to the matrix `M`. Preserves sparsity.
"""
function applymap(Φ::AbstractMatrix{T}, M::AbstractMatrix{S}) where {T,S}
    din = size(M, 1)
    dtotal = size(Φ, 1)
    dout = dtotal ÷ din
    @assert dtotal == din * dout
    TS = Base.promote_op(*, T, S)
    if SA.issparse(Φ) && SA.issparse(M)
        result = SA.spzeros(TS, dout, dout)
    else
        result = Matrix{TS}(undef, dout, dout)
    end
    applymap!(result, Φ, M)
    return _wrapper_applymap(M, T)(result)
end

@doc """
     applymap!(result::AbstractMatrix, Φ::AbstractMatrix, M::AbstractMatrix)

Applies the CP map given by the Choi-Jamiołkowski operator `Φ` to the matrix `M` without allocating or wrapping. In the symmetric or Hermitian cases only the upper triangular is computed. `result` must be a matrix of size `dout × dout`,  where `size(M, 1) * dout == size(Φ, 1)`.
""" applymap!(result::AbstractMatrix, Φ::AbstractMatrix, M::AbstractMatrix)

for (matrixtype, limit) ∈ ((:AbstractMatrix, :dout), (:Symmetric, :j), (:Hermitian, :j))
    @eval begin
        function applymap!(result::AbstractMatrix, Φ::AbstractMatrix, M::$matrixtype)
            din = size(M, 1)
            dtotal = size(Φ, 1)
            dout = dtotal ÷ din
            @assert dtotal == din * dout
            @inbounds for j ∈ 1:dout, i ∈ 1:$limit
                result[i, j] = 0
                for l ∈ 1:din, k ∈ 1:din
                    result[i, j] += M[k, l] * Φ[(k-1)*dout+i, (l-1)*dout+j]
                end
            end
            return result
        end
    end
end

"""
    channel_bit_flip(p::Real)

Return the Kraus operator representation of the bit flip channel. It applies Pauli-X with probability `1 − p` (flip from |0⟩ to |1⟩ and vice versa).
"""
function channel_bit_flip(p::Real)
    E0 = [sqrt(p) 0; 0 sqrt(p)]
    E1 = [0 sqrt(1 - p); sqrt(1 - p) 0]
    return [E0, E1]
end
export channel_bit_flip

"""
    channel_phase_flip(p::Real)

Return the Kraus operator representation of the phase flip channel. It applies Pauli-Z with probability `1 − p`.
"""
function channel_phase_flip(p::Real)
    E0 = [sqrt(p) 0; 0 sqrt(p)]
    E1 = [sqrt(1 - p) 0; 0 -sqrt(1 - p)]
    return [E0, E1]
end
export channel_phase_flip

"""
    channel_bit_phase_flip(p::Real)

Return the Kraus operator representation of the phase flip channel. It applies Pauli-Y (=iXY) with probability `1 − p`.
"""
function channel_bit_phase_flip(p::Real)
    E0 = [sqrt(p) 0; 0 sqrt(p)]
    E1 = [0 -im*sqrt(1 - p); im*sqrt(1 - p) 0]
    return [E0, E1]
end
export channel_bit_phase_flip

"""
    channel_depolarizing(v::Real, d::Integer = 2)

Return the Kraus operator representation of the depolarizing channel of dimension `d`. It replaces a single qudit by the completely mixed state with probability '1-v'.
"""
function channel_depolarizing(v::Real, d::Integer = 2)
    K = [zeros(typeof(v), d, d) for _ ∈ 1:d^2+1]
    rootv = sqrt(v)
    for i ∈ 1:d
        K[1][i, i] = rootv
    end
    rootvd = sqrt((1 - v) / d)
    for j ∈ 1:d, i ∈ 1:d
        K[i+(j-1)*d+1][i, j] = rootvd
    end
    return K
end
export channel_depolarizing

"""
    channel_amplitude_damping(γ::Real)

Return the Kraus operator representation of the amplitude damping channel.
It describes the effect of dissipation to an environment at zero temperature.
`γ` is the probability of the system to decay to the ground state.
"""
function channel_amplitude_damping(γ::Real)
    E0 = [1 0; 0 sqrt(1 - γ)]
    E1 = [0 sqrt(γ); 0 0]
    return [E0, E1]
end
export channel_amplitude_damping

"""
    channel_amplitude_damping_generalized(rho::AbstractMatrix, p::Real, γ::Real)

Return the Kraus operator representation of the generalized amplitude damping channel.
It describes the effect of dissipation to an environment at finite temperature.
`γ` is the probability of the system to decay to the ground state.
`1-p` can be thought as the energy of the stationary state.
"""
function channel_amplitude_damping_generalized(p::Real, γ::Real)
    E0 = [sqrt(p) 0; 0 sqrt(p)*sqrt(1 - γ)]
    E1 = [0 sqrt(p)*sqrt(γ); 0 0]
    E2 = [sqrt(1 - p)*sqrt(1 - γ) 0; 0 sqrt(1 - p)]
    E3 = [0 0; sqrt(1 - p)*sqrt(γ) 0]
    return [E0, E1, E2, E3]
end
export channel_amplitude_damping_generalized

"""
    channel_phase_damping(λ::Real)
    
Return the Kraus operator representation of the phase damping channel. It describes the photon scattering or electron perturbation. 'λ' is the probability being scattered or perturbed (without loss of energy).
"""
function channel_phase_damping(λ::Real) # It can be reformulated as channel_phase_flip(rho, p = (1+√(1 − λ))/2)
    E0 = [1 0; 0 sqrt(1 - λ)]
    E1 = [0 0; 0 sqrt(λ)]
    return [E0, E1]
end
export channel_phase_damping

"""
    choi(K::Vector{<:AbstractMatrix})

Constructs the Choi-Jamiołkowski representation of the CP map given by the Kraus operators `K`. Preserves sparsity.
The convention used is that choi(K) = ∑ᵢⱼ |i⟩⟨j|⊗K|i⟩⟨j|K'.
"""
function choi(K::Vector{<:AbstractMatrix})
    if all(SA.issparse.(K))
        vecK = SA.SparseVector.(vec.(K))
        result = vecK[1] * vecK[1]'
        for i ∈ 2:length(vecK)
            result .+= vecK[i] * vecK[i]'
        end
    else
        d = length(K[1])
        result = vec(K[1]) * vec(K[1])'
        @inbounds for k ∈ 2:length(K)
            for j ∈ 1:d
                for i ∈ 1:j
                    result[i, j] += K[k][i] * conj(K[k][j])
                end
            end
        end
    end
    return Hermitian(result)
end
export choi

"""
    diamond_norm(
        J::AbstractMatrix,
        dims::AbstractVecOrTuple;
        verbose::Bool = false,
        solver = Hypatia.Optimizer{_solver_type(T)})

Computes the diamond norm of the supermap `J` given in the Choi-Jamiołkowski representation, with subsystem dimensions `dims`.

Reference: [Diamond norm](https://en.wikipedia.org/wiki/Diamond_norm)
"""
function diamond_norm(
    J::AbstractMatrix{T},
    dims::AbstractVecOrTuple;
    verbose = false,
    solver = Hypatia.Optimizer{_solver_type(T)}
) where {T}
    ishermitian(J) || throw(ArgumentError("Supermap needs to be Hermitian"))

    is_complex = T <: Complex
    psd_cone, wrapper, hermitian_space = _sdp_parameters(is_complex)
    din, dout = dims
    model = JuMP.GenericModel{_solver_type(T)}()
    JuMP.@variable(model, Y[1:din*dout, 1:din*dout] ∈ hermitian_space)
    JuMP.@variable(model, σ[1:din, 1:din] ∈ hermitian_space)
    bigσ = wrapper(kron(σ, I(dout)))
    JuMP.@constraint(model, bigσ - Y ∈ psd_cone)
    JuMP.@constraint(model, bigσ + Y ∈ psd_cone)

    JuMP.@constraint(model, tr(σ) == 1)
    JuMP.@objective(model, Max, real(dot(J, Y)))

    JuMP.set_optimizer(model, solver)
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model) || error(JuMP.raw_status(model))
    return JuMP.objective_value(model)
end
export diamond_norm

"""
    diamond_norm(K::Vector{<:AbstractMatrix})

Computes the diamond norm of the CP map given by the Kraus operators `K`.
"""
function diamond_norm(K::Vector{<:AbstractMatrix})
    dual_to_id = sum(Hermitian(Ki' * Ki) for Ki ∈ K)
    return opnorm(dual_to_id)
end
