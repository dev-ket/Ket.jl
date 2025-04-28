#extract from T the kind of float to be used in the conic solver
_solver_type(::Type{T}) where {T<:AbstractFloat} = T
_solver_type(::Type{Complex{T}}) where {T<:AbstractFloat} = T
_solver_type(::Type{T}) where {T<:Number} = Float64

@doc """
    applykraus(K::Vector{<:AbstractMatrix}, M::AbstractMatrix)

Applies the CP map given by the Kraus operators `K` to the matrix `M`.
""" applykraus(K::Vector{<:AbstractMatrix{T}}, M::AbstractMatrix{S}) where {T,S}

for (matrixtype, wrapper) ∈ ((:AbstractMatrix, :identity), (:Symmetric, :Symmetric), (:Hermitian, :Hermitian))
    @eval begin
        function applykraus(K::Vector{<:AbstractMatrix{T}}, M::$matrixtype{S}) where {T,S}
            dout, din = size(K[1])
            TS = Base.promote_op(*, T, S)
            temp = Matrix{TS}(undef, dout, din)
            result = Matrix{TS}(undef, dout, dout)
            return $wrapper(applykraus!(result, K, M, temp))
        end
    end
end
export applykraus

"""
    applykraus!(result::Matrix, K::Vector{<:AbstractMatrix}, M::AbstractMatrix, temp::Matrix)

Applies the CP map given by the Kraus operators `K` to the matrix `M` without allocating. `result` and `temp` must be
matrices of size `dout × dout` and `dout × din`, where `dout, din == size(K[1])`.
"""
function applykraus!(result::Matrix, K::Vector{<:AbstractMatrix}, M::AbstractMatrix, temp::Matrix)
    mul!(temp, K[1], M)
    mul!(result, temp, K[1]')
    for i ∈ 2:length(K)
        mul!(temp, K[i], M)
        mul!(result, temp, K[i]', true, true)
    end
    return result
end
export applykraus!

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
    for i = 1:d
        K[1][i, i] = rootv
    end
    rootvd = sqrt((1-v)/d)
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

Constructs the Choi-Jamiołkowski representation of the CP map given by the Kraus operators `K`.
The convention used is that choi(K) = ∑ᵢⱼ |i⟩⟨j|⊗K|i⟩⟨j|K'
"""
choi(K::Vector{<:AbstractMatrix}) = sum(ketbra(vec(Ki)) for Ki ∈ K)
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
function diamond_norm(J::AbstractMatrix{T}, dims::AbstractVecOrTuple; verbose = false, solver = Hypatia.Optimizer{_solver_type(T)}) where {T}
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
    JuMP.is_solved_and_feasible(model) || throw(error(JuMP.raw_status(model)))
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

# Construct the Choi operator of an Amplitude Damping channel 
function channel_discrimination()
    γ=67/100
    # Write a Cell with the Kraus operators
    # Since this map is Completely positive, left and right Choi operators are the same.
    K0=[1 0; 0 sqrt(1-γ)]
    K1 = [0 sqrt(γ); 0 0]
    # Declare the Channels which will be used
    C1= choi([K0,K1]);

    # Construct the Choi operator of an Bit Flip channel 
    η=87/100
    K0=sqrt(η)* I(2)
    K1=sqrt(1-η)*[0 1; 1 0]

    C2 = choi([K0,K1])

    C = Array{ComplexF64}(undef, 4, 4, 2)  # Création d’un tableau 3D vide (2×2×2)

    C[:, :, 1] = C1
    C[:, :, 2] = C2

    N=size(C,3); #Obtain the number of channels N
    k=2; #Set the number of uses k equals 2

    d=Int(sqrt(size(C[:,:,1],1)));
    dIn=d;
    dOut=d;
    DIM=[d,d,d,d];
    p_i=ones(1,N)/N;
    println(d ," ", p_i)

    model = JuMP.Model(() -> Hypatia.Optimizer(verbose = false))
    T = [JuMP.@variable(model, [1:Int(dIn^(2*k)), 1:Int(dOut^(2*k))] in JuMP.HermitianPSDCone()) for i in 1:N]
    JuMP.@constraint(model, partial_trace(sum(T), [1]) == I)
    JuMP.@constraint(model, trace_replace(sum(T), [2,4],DIM) == sum(T))
    JuMP.@constraint(model, tr((sum(T))) == dOut ^2)

    JuMP.@objective(
        model,
        Max,
        sum([real(LinearAlgebra.tr(p_i[i]* T[i] * kron(fill(C[:,:,i],k)...))) for i in 1:N])
    )

    JuMP.optimize!(model)
    JuMP.assert_is_solved_and_feasible(model)
    JuMP.solution_summary(model)

    Tsolution = [JuMP.value.(t) for t in T]
    println(JuMP.objective_value(model))
    return Tsolution
end
export channel_discrimination