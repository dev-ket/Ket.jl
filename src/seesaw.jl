"""
    seesaw(
        CG::AbstractArray,
        scenario::Tuple,
        d::Integer,
        n_trials::Integer = 1;
        verbose::Bool = false,
        solver = Hypatia.Optimizer{_solver_type(T)},
        method::Symbol = :auto)


Maximizes a N-partite Bell functional `CG` in Collins-Gisin notation using the seesaw heuristic.
`scenario` is a vector detailing the number of inputs and outputs, in the order (`o1`, .... , `oN`, `i1`, ..., `iN`).
`d` is an integer determining the local dimension of the strategy.

Returns a tuple ω, ψ, all_measurements where ω is the maximum found, ψ the state,
and all_measurement a list containing the POVMs of each party.

If all outputs equal 2 the heuristic reduces to a bunch of eigenvalue problems.
Otherwise semidefinite programming is needed and we use the assemblage version of seesaw.

`method` controls which algorithm is used: `:auto` (default) selects eigenvalue when all outputs equal 2
and SDP otherwise; `:eigenvalue` forces the eigenvalue path (requires all outputs equal 2);
`:sdp` forces the SDP path regardless of the output cardinalities.

`verbose` prints solver output when `true`. `solver` overrides the default conic solver.

The heuristic is executed `n_trials` times, and the best result is returned.

References:
- Pál and Vértesi, [arXiv:1006.3032](https://arxiv.org/abs/1006.3032)
- Tavakoli et al., [arXiv:2307.02551](https://arxiv.org/abs/2307.02551) (Sec. II.B.1)
"""
function seesaw(
    CG::Array{T,N},
    scenario::Tuple,
    d::Integer,
    n_trials::Integer = 1;
    verbose = false,
    solver = Hypatia.Optimizer{_solver_type(T)},
    method::Symbol = :auto
) where {T<:Real,N}
    @assert length(scenario) == 2N
    @assert method ∈ (:auto, :eigenvalue, :sdp)
    binary_outputs = all(scenario[1:N] .== 2)
    method == :eigenvalue && !binary_outputs && throw(ArgumentError("method = :eigenvalue requires all outputs to equal 2"))
    use_eigenvalue = method == :eigenvalue || (method == :auto && binary_outputs)
    R = _solver_type(T)
    CG = R.(CG)
    minimumincrease = _rtol(R)
    maxiter = 100
    ω0 = typemin(R)
    local ψ0, all_measurements

    for _ ∈ 1:n_trials
        if use_eigenvalue
            ω, ψ, temp_measurements = _seesaw_eigenvalue(CG, d, minimumincrease, maxiter)
        else
            ω, ψ, temp_measurements = _seesaw_sdp(CG, scenario, d, minimumincrease, maxiter; verbose, solver)
        end
        if ω > ω0
            ω0, ψ0, all_measurements = ω, ψ, temp_measurements
        end
    end

    for party_povms ∈ all_measurements
        for POVM ∈ party_povms
            last_element = I - sum(POVM)
            push!(POVM, last_element)
        end
    end
    return ω0, ψ0, all_measurements...
end
export seesaw

function _seesaw_sdp(CG::Array{R,N}, scenario, d, minimumincrease, maxiter; verbose, solver) where {R<:AbstractFloat,N}
    outs = scenario[1:N]
    ins = scenario[N+1:2N]
    T2 = Complex{R}

    all_povms = [[random_povm(T2, d, outs[n])[1:outs[n]-1] for _ ∈ 1:ins[n]] for n ∈ 2:N]

    local ψ0, all_measurements0
    ω0 = typemin(R)
    i = 0
    while true
        i += 1
        ω, ρxa, ρ_rest = _optimize_assemblage(CG, scenario, all_povms; verbose, solver)
        for k ∈ 1:N-1
            ω, all_povms[k] = _optimize_party_povm(CG, scenario, k + 1, ρxa, ρ_rest, all_povms; verbose, solver)
        end
        if ω - ω0 ≤ minimumincrease || i > maxiter
            ω0 = ω
            ψ0, M1 = _decompose_assemblage(scenario, d, ρxa, ρ_rest)
            all_measurements0 = pushfirst!(copy(all_povms), M1)
            break
        end
        ω0 = ω
    end
    return ω0, ψ0, all_measurements0
end

function _optimize_assemblage(
    CG::Array{R,N},
    scenario,
    all_povms;
    verbose = false,
    solver = Hypatia.Optimizer{R}
) where {R<:AbstractFloat,N}
    outs = scenario[1:N]
    ins = scenario[N+1:2N]
    d = size(all_povms[1][1][1], 1)
    Dp = d^(N - 1)
    T2 = Complex{R}

    model = JuMP.GenericModel{R}()
    ρxa = [[JuMP.@variable(model, [1:Dp, 1:Dp] ∈ JuMP.HermitianPSDCone()) for _ ∈ 1:outs[1]-1] for _ ∈ 1:ins[1]]
    ρ_rest = JuMP.@variable(model, [1:Dp, 1:Dp], Hermitian)

    JuMP.@constraint(model, tr(ρ_rest) == 1)
    for x ∈ 1:ins[1]
        JuMP.@constraint(model, ρ_rest - sum(ρxa[x][a] for a ∈ 1:outs[1]-1) ∈ JuMP.HermitianPSDCone())
    end

    ω = _compute_value_assemblage(CG, scenario, ρxa, ρ_rest, all_povms)
    JuMP.@objective(model, Max, ω)

    JuMP.set_optimizer(model, solver)
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model) || @warn JuMP.raw_status(model)
    value_ρxa = [[JuMP.value(ρxa[x][a]) for a ∈ 1:outs[1]-1] for x ∈ 1:ins[1]]
    value_ρ_rest = JuMP.value(ρ_rest)
    return JuMP.value(ω)::R, value_ρxa, value_ρ_rest
end

# Returns the operator for party p+1 at Collins-Gisin index cn, decoding (a, x) from cn:
# identity for marginal (cn=1),
# POVM element all_povms[p][x][a] otherwise
function _cg_op(all_povms, p, cn, o_p, Id)
    cn == 1 && return Id
    a = (cn - 2) % (o_p - 1) + 1
    x = (cn - 2) ÷ (o_p - 1) + 1
    return Matrix(all_povms[p][x][a])
end

# Returns the assemblage element for party 1 at Collins-Gisin index c1:
# ρ_rest for marginal (c1=1),
# ρxa[x][a] otherwise.
function _cg_state(c1, o1, ρxa, ρ_rest)
    c1 == 1 && return ρ_rest
    a = (c1 - 2) % (o1 - 1) + 1
    x = (c1 - 2) ÷ (o1 - 1) + 1
    return ρxa[x][a]
end

# Precomputes d×d effective operators Γ[xk][ak] and a scalar constant such that
# the Bell value equals constant + Σ_{xk,ak} tr(Mk[xk][ak] * Γ[xk][ak]).
function _effective_operators(CG::Array{R,N}, scenario, party_k, ρxa, ρ_rest, all_povms) where {R<:AbstractFloat,N}
    outs = scenario[1:N]
    ins = scenario[N+1:2N]
    o1, ok, ik = outs[1], outs[party_k], ins[party_k]
    d = size(all_povms[1][1][1], 1)
    T2 = Complex{R}
    Id = Matrix{T2}(I, d, d)

    k_rest = party_k - 1
    remove = collect(setdiff(1:N-1, k_rest))
    dims_rest = fill(d, N - 1)

    Γ = [[zeros(T2, d, d) for _ ∈ 1:ok-1] for _ ∈ 1:ik]
    constant = zero(R)

    for ci ∈ CartesianIndices(size(CG))
        coeff = CG[ci]
        iszero(coeff) && continue

        state = _cg_state(ci[1], o1, ρxa, ρ_rest)
        ops = [p == k_rest ? Id : _cg_op(all_povms, p, ci[p+1], outs[p+1], Id) for p ∈ 1:N-1]
        O_others = length(ops) == 1 ? ops[1] : kron(ops...)
        M = partial_trace(state * O_others, remove, dims_rest)

        ck = ci[party_k]
        if ck == 1
            constant += coeff * real(tr(M))
        else
            ak = (ck - 2) % (ok - 1) + 1
            xk = (ck - 2) ÷ (ok - 1) + 1
            Γ[xk][ak] .+= coeff .* M
        end
    end

    return Γ, constant
end

function _compute_value_assemblage(CG::Array{R,N}, scenario, ρxa, ρ_rest, all_povms) where {R<:AbstractFloat,N}
    outs = scenario[1:N]
    ins = scenario[N+1:2N]
    o1 = outs[1]
    cgind(a, x, o) = 1 + a + (x - 1) * (o - 1)
    d = size(all_povms[1][1][1], 1)
    Id = Matrix{Complex{R}}(I, d, d)

    rest_sizes = Tuple(1 + ins[p+1] * (outs[p+1] - 1) for p ∈ 1:N-1)

    ω = zero(JuMP.GenericAffExpr{R,JuMP.GenericVariableRef{R}})

    for ct ∈ CartesianIndices(rest_sizes)
        O_rest = _cg_op(all_povms, 1, ct[1], outs[2], Id)
        for p ∈ 2:N-1
            O_rest = kron(O_rest, _cg_op(all_povms, p, ct[p], outs[p+1], Id))
        end

        # party 1 marginal (c1 = 1)
        coeff = CG[1, ct.I...]
        if !iszero(coeff)
            JuMP.add_to_expression!(ω, coeff, real(dot(O_rest, ρ_rest)))
        end

        # party 1 non-marginal terms
        for x ∈ 1:ins[1], a ∈ 1:o1-1
            coeff = CG[cgind(a, x, o1), ct.I...]
            if !iszero(coeff)
                JuMP.add_to_expression!(ω, coeff, real(dot(O_rest, ρxa[x][a])))
            end
        end
    end

    return ω
end

function _optimize_party_povm(
    CG::Array{R,N},
    scenario,
    party_k,
    ρxa,
    ρ_rest,
    all_povms;
    verbose = false,
    solver = Hypatia.Optimizer{R}
) where {R<:AbstractFloat,N}
    ok = scenario[party_k]
    ik = scenario[N + party_k]
    d = size(all_povms[1][1][1], 1)

    Γ, constant = _effective_operators(CG, scenario, party_k, ρxa, ρ_rest, all_povms)

    # SDP with d×d POVM variables
    model = JuMP.GenericModel{R}()
    Mk = [[JuMP.@variable(model, [1:d, 1:d] ∈ JuMP.HermitianPSDCone()) for _ ∈ 1:ok-1] for _ ∈ 1:ik]
    for xk ∈ 1:ik
        JuMP.@constraint(model, I - sum(Mk[xk][ak] for ak ∈ 1:ok-1) ∈ JuMP.HermitianPSDCone())
    end

    ω = constant * one(JuMP.GenericAffExpr{R,JuMP.GenericVariableRef{R}})
    for xk ∈ 1:ik, ak ∈ 1:ok-1
        JuMP.add_to_expression!(ω, 1, real(dot(Γ[xk][ak], Mk[xk][ak])))
    end
    JuMP.@objective(model, Max, ω)

    JuMP.set_optimizer(model, solver)
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model) || @warn JuMP.raw_status(model)
    value_Mk = [[JuMP.value(Mk[xk][ak]) for ak ∈ 1:ok-1] for xk ∈ 1:ik]
    return JuMP.value(ω)::R, value_Mk
end

# Extracts state ψ ∈ C^{d·Dp} and party 1's d×d POVMs from the assemblage.
# Eigendecomposes ρ_rest, caps rank at d (party 1's local dimension).
function _decompose_assemblage(scenario, d::Integer, ρxa, ρ_rest::AbstractMatrix{T}) where {T}
    N = length(scenario) ÷ 2
    o1, i1 = scenario[1], scenario[N+1]
    Dp = size(ρ_rest, 1)

    λ, U = eigen(ρ_rest) # ascending eigenvalues
    r = min(d, count(x -> x ≥ _rtol(T), λ))
    top = (Dp - r + 1):Dp # indices of r largest eigenvalues

    ψ = zeros(T, d * Dp)
    for (i, t) ∈ enumerate(top)
        ψ[(i-1)*Dp+1:i*Dp] = sqrt(λ[t]) * U[:, t]
    end

    invrootλ = map(x -> x ≥ _rtol(T) ? 1 / sqrt(x) : zero(x), λ[top])
    V = U[:, top] * Diagonal(invrootλ) # Dp×r
    M1 = Vector{Measurement{T}}(undef, i1)
    for x ∈ 1:i1
        M1[x] = Measurement{T}(undef, o1 - 1)
        for a ∈ 1:o1-1
            small = Hermitian(conj(V' * ρxa[x][a] * V)) # r×r
            padded = zeros(T, d, d)
            padded[1:r, 1:r] = small
            M1[x][a] = Hermitian(padded)
        end
    end
    return ψ, M1
end

# Both eigenvalue seesaw, with different signatures for bipartite (and schmidt optimization) and multipartite

function _seesaw_eigenvalue(CG::Array{R,N}, d, minimumincrease, maxiter) where {R<:AbstractFloat,N}
    T2 = Complex{R}
    dims = fill(d, N)
    D = d^N
    ψ = normalize!(complex.(randn(R, D), randn(R, D)))
    Ms = [[random_povm(T2, d, 2)[1] for _ ∈ 1:size(CG, k)-1] for k ∈ 1:N]
    ω0 = typemin(R)
    local ψ0, Ms0
    i = 0
    while true
        i += 1
        ρ = ψ * ψ'
        for k ∈ 1:N
            _optimize_multi_projectors!(CG, ρ, Ms, k, dims)
        end
        ω, ψ = _optimize_multi_state!(CG, Ms, dims)
        if ω - ω0 ≤ minimumincrease || i > maxiter
            ω0 = ω
            ψ0 = ψ
            Ms0 = deepcopy(Ms)
            break
        end
        ω0 = ω
    end
    all_measurements = [[[Ms0[k][xk]] for xk ∈ 1:size(CG, k)-1] for k ∈ 1:N]
    return ω0, ψ0, all_measurements
end

function _seesaw_eigenvalue(CG::Matrix{R}, d, minimumincrease, maxiter) where {R<:AbstractFloat}
    ia, ib = size(CG) .- 1
    T2 = Complex{R}
    λ = T2.(sqrt.(random_probability(R, d)))
    B = [random_povm(T2, d, 2)[1] for _ ∈ 1:ib]::Measurement{T2}
    A = [Hermitian(zeros(T2, d, d)) for _ ∈ 1:ia]::Measurement{T2}
    local ψ0, A0, B0
    ω0 = typemin(R)
    i = 0
    while true
        i += 1
        _optimize_alice_projectors!(CG, λ, A, B)
        _optimize_bob_projectors!(CG, λ, A, B)
        ω = _optimize_state!(CG, λ, A, B)
        if ω - ω0 ≤ minimumincrease || i > maxiter
            ω0 = ω
            ψ0 = state_phiplus_ket(T2, d; coeff = λ)
            A0 = [[A[x]] for x ∈ 1:ia] # rather inconvenient format
            B0 = [[B[y]] for y ∈ 1:ib] # but consistent with the general case
            break
        end
        ω0 = ω
    end
    return ω0, ψ0, [A0, B0]
end

# Multipartite eigenvalues functions 
function _optimize_multi_projectors!(CG::Array{R,N}, ρ, Ms, k, dims) where {R<:AbstractFloat,N}
    d = dims[k]
    T2 = Complex{R}
    Id = Matrix{T2}(I, d, d)
    remove = collect(setdiff(1:N, k))
    ins_k = size(CG, k) - 1
    for xk ∈ 1:ins_k
        Γ = zeros(T2, d, d)
        for ci ∈ CartesianIndices(size(CG))
            ci[k] == xk + 1 || continue
            coeff = CG[ci]
            iszero(coeff) && continue
            O = kron([j == k ? Id : (ci[j] == 1 ? Id : Ms[j][ci[j]-1]) for j ∈ 1:N]...)
            Γ .+= coeff .* partial_trace(ρ * O, remove, dims)
        end
        Γh = Hermitian(Γ)
        _positive_projection!(Γh)
        Ms[k][xk] = Γh
    end
end

function _optimize_multi_state!(CG::Array{R,N}, Ms, dims) where {R<:AbstractFloat,N}
    d = dims[1]
    D = prod(dims)
    T2 = Complex{R}
    Id = Matrix{T2}(I, d, d)
    W = zeros(T2, D, D)
    for ci ∈ CartesianIndices(size(CG))
        coeff = CG[ci]
        iszero(coeff) && continue
        ops = [ci[k] == 1 ? Id : Ms[k][ci[k]-1] for k ∈ 1:N]
        W .+= coeff .* kron(ops...)
    end
    vals, U = eigen(Hermitian(W))
    return real(vals[end])::R, U[:, end]
end

# Bipartite eigenvalues functions 

function _optimize_alice_projectors!(CG::Matrix, λ::Vector, A, B)
    ia, ib = size(CG) .- 1
    d = length(λ)
    for x ∈ 1:ia
        for j ∈ 1:d, i ∈ 1:j
            A[x].data[i, j] = CG[x+1, 1] * (i == j) * abs2(λ[i])
            for y ∈ 1:ib
                A[x].data[i, j] += CG[x+1, y+1] * λ[i] * conj(λ[j] * B[y][i, j])
            end
        end
        _positive_projection!(A[x])
    end
end

function _optimize_bob_projectors!(CG::Matrix, λ::Vector, A, B)
    ia, ib = size(CG) .- 1
    d = length(λ)
    for y ∈ 1:ib
        for j ∈ 1:d, i ∈ 1:j
            B[y].data[i, j] = CG[1, y+1] * (i == j) * abs2(λ[i])
            for x ∈ 1:ia
                B[y].data[i, j] += CG[x+1, y+1] * λ[i] * conj(λ[j] * A[x][i, j])
            end
        end
        _positive_projection!(B[y])
    end
end

function _positive_projection!(M::AbstractMatrix{T}) where {T}
    λ, U = eigen(M)
    fill!(M.data, 0)
    temp = similar(M.data)
    for i ∈ 1:length(λ)
        if λ[i] > _rtol(T)
            @views mul!(temp, U[:, i], U[:, i]') #5-argument mul! doesn't call herk for Julia ≤ 1.11
            M.data .+= temp
        end
    end
    return M
end

function _optimize_state!(CG::Matrix, λ::Vector{T}, A, B) where {T}
    ia, ib = size(CG) .- 1
    d = length(λ)
    M = Hermitian(zeros(T, d, d))
    for x ∈ 1:ia
        M += CG[x+1, 1] * real(Diagonal(A[x]))
    end
    for y ∈ 1:ib
        M += CG[1, y+1] * real(Diagonal(B[y]))
    end
    for y ∈ 1:ib, x ∈ 1:ia, j ∈ 1:d, i ∈ 1:j
        M.data[i, j] += CG[x+1, y+1] * A[x].data[i, j] * B[y].data[i, j]
    end
    vals, U = eigen!(M)
    λ .= U[:, d]
    return CG[1, 1] + vals[d]
end
