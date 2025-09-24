"""
    tsirelson_bound(CG::Array, scenario::Tuple, level; verbose::Bool = false, dualize::Bool = false, solver = Hypatia.Optimizer{_solver_type(T)})

Upper bounds the Tsirelson bound of a multipartite Bell funcional `CG`, written in Collins-Gisin notation.
`scenario` is a tuple detailing the number of inputs and outputs, in the order (oa, ob, ..., ia, ib, ...).
`level` is an integer or a string like "1 + A B" determining the level of the NPA hierarchy.
`verbose` determines whether solver output is printed.
`dualize` determines whether the dual problem is solved instead. WARNING: This is critical for performance, and the correct choice depends on the solver.
"""
function tsirelson_bound(
    CG::Array{T,N},
    scenario::Tuple,
    level;
    verbose::Bool = false,
    dualize::Bool = false,
    solver = Hypatia.Optimizer{_solver_type(T)}
) where {T<:Number,N}
    @assert length(scenario) == 2N
    CG = _solver_type(T).(CG)
    if N == 2
        if level == 1
            return _tsirelson_bound_manual(CG, scenario, false; verbose, dualize = !dualize, solver)
        elseif level == "1 + A B" || level == "1+ A B" || level == "1 +A B" || level == "1+A B"
            if max(scenario[3], scenario[4]) ≥ 3 #heuristic for when it's faster
                return _tsirelson_bound_manual(CG, scenario, true; verbose, dualize = !dualize, solver)
            end
        end
    end
    outs = scenario[1:N]
    ins = scenario[N+1:2N]
    Π = [[[QuantumNPA.Id for _ ∈ 1:outs[n]-1] QuantumNPA.projector(n, 1:outs[n]-1, 1:ins[n])] for n ∈ 1:N]
    cgindex(a, x) = (x .!= 1) .* (a .+ (x .- 2) .* (outs .- 1)) .+ 1
    behaviour_operator = Array{QuantumNPA.Monomial,N}(undef, size(CG))
    for x ∈ CartesianIndices(ins .+ 1)
        for a ∈ CartesianIndices((x.I .!= 1) .* (outs .- 2) .+ 1)
            behaviour_operator[cgindex(a.I, x.I)...] = prod(Π[n][a.I[n], x.I[n]] for n ∈ 1:N)
        end
    end
    Q, behaviour = _npa(CG, behaviour_operator, level; verbose, solver, dualize)
    return Q, behaviour
end
export tsirelson_bound

"""
    tsirelson_bound(FC::Array, level; verbose::Bool = false, dualize::Bool = false, solver = Hypatia.Optimizer{_solver_type(T)})

Upper bounds the Tsirelson bound of a multipartite Bell funcional `FC`, written in correlation notation.
`level` is an integer or a string like "1 + A B" determining the level of the NPA hierarchy.
`verbose` determines whether solver output is printed.
`dualize` determines whether the dual problem is solved instead. WARNING: This is critical for performance, and the correct choice depends on the solver.
"""
function tsirelson_bound(
    FC::Array{T,N},
    level;
    verbose::Bool = false,
    dualize::Bool = false,
    solver = Hypatia.Optimizer{_solver_type(T)}
) where {T<:Number,N}
    FC = _solver_type(T).(FC)
    if N == 2
        if level == 1
            return _tsirelson_bound_manual(FC, false; verbose, dualize = !dualize, solver)
        elseif level == "1 + A B" || level == "1+ A B" || level == "1 +A B" || level == "1+A B"
            return _tsirelson_bound_manual(FC, true; verbose, dualize = !dualize, solver)
        end
    end
    ins = size(FC) .- 1
    O = [[QuantumNPA.Id; QuantumNPA.dichotomic(n, 1:ins[n])] for n ∈ 1:N]

    behaviour_operator = Array{QuantumNPA.Monomial,N}(undef, size(FC))
    for x ∈ CartesianIndices(ins .+ 1)
        behaviour_operator[x] = prod(O[n][x[n]] for n ∈ 1:N)
    end

    Q, behaviour = _npa(FC, behaviour_operator, level; verbose, solver, dualize)
    return Q, behaviour
end

function _npa(functional::Array{T,N}, behaviour_operator, level; verbose, solver, dualize) where {T<:AbstractFloat,N}
    model = JuMP.GenericModel{T}()
    Γ_basis = QuantumNPA.npa_moment(behaviour_operator, level)
    monomials = setdiff(QuantumNPA.monomials(Γ_basis), [QuantumNPA.Id])
    JuMP.@variable(model, var[monomials])
    dΓ = size(Γ_basis)[1]
    Γ = Matrix{typeof(1 * first(var))}(undef, dΓ, dΓ)
    for i ∈ eachindex(Γ)
        Γ[i] = 0
    end
    Γ .+= Γ_basis[QuantumNPA.Id]
    for m ∈ monomials
        _jump_muladd!(Γ, Γ_basis[m], var[m])
    end
    JuMP.@constraint(model, Γ ∈ JuMP.PSDCone())
    behaviour = Array{typeof(1 * first(var)),N}(undef, size(behaviour_operator))
    behaviour[1] = 1
    for i ∈ 2:length(behaviour)
        behaviour[i] = var[behaviour_operator[i]]
    end
    objective = dot(functional, behaviour)
    JuMP.@objective(model, Max, objective)
    if dualize
        JuMP.set_optimizer(model, Dualization.dual_optimizer(solver; coefficient_type = T))
    else
        JuMP.set_optimizer(model, solver)
    end
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model) || @warn JuMP.raw_status(model)
    return JuMP.objective_value(model)::T, JuMP.value.(behaviour)::Array{T,N}
end

function _jump_muladd!(G, A::SA.SparseMatrixCSC, jumpvar)
    for j ∈ 1:size(A, 2)
        for k ∈ SA.nzrange(A, j)
            JuMP.add_to_expression!(G[SA.rowvals(A)[k], j], SA.nonzeros(A)[k], jumpvar)
        end
    end
    return G
end

function _tsirelson_bound_manual(
    CG::Matrix{T},
    scenario::Tuple,
    include_ab::Bool;
    verbose,
    dualize,
    solver
) where {T<:AbstractFloat}
    oa, ob, ia, ib = scenario
    alice_ops = ia * (oa - 1)
    bob_ops = ib * (ob - 1)
    dq1 = 1 + alice_ops + bob_ops
    dq1ab = dq1 + alice_ops * bob_ops
    model = JuMP.GenericModel{T}()
    dΓ = include_ab ? dq1ab : dq1
    JuMP.@variable(model, Γ[1:dΓ, 1:dΓ] in JuMP.PSDCone())
    ## normalization constraints
    JuMP.@constraint(model, Γ[1, 1] == 1)
    for i ∈ 2:dq1
        JuMP.@constraint(model, Γ[1, i] == Γ[i, i])
    end
    ## q1 orthogonality constraints
    aind(a, x) = 1 + a + (x - 1) * (oa - 1)
    for x ∈ 1:ia, a1 ∈ 1:oa-1, a2 ∈ a1+1:oa-1
        JuMP.@constraint(model, Γ[aind(a1, x), aind(a2, x)] == 0)
    end
    bind(b, y) = 1 + alice_ops + b + (y - 1) * (ob - 1)
    for y ∈ 1:ib, b1 ∈ 1:ob-1, b2 ∈ b1+1:ob-1
        JuMP.@constraint(model, Γ[bind(b1, y), bind(b2, y)] == 0)
    end

    alice_marginal = Γ[1, 2:alice_ops+1]
    bob_marginal = Γ[1, alice_ops+2:dq1]
    correlation = Γ[2:alice_ops+1, alice_ops+2:dq1]
    behaviour = [Γ[1, 1] bob_marginal'; alice_marginal correlation]

    if include_ab
        ## first line of q1ab
        JuMP.@constraint(model, Γ[1, dq1+1:dq1ab] .== vec(correlation'))

        ## more normalization
        for i ∈ dq1+1:dq1ab
            JuMP.@constraint(model, Γ[1, i] == Γ[i, i])
        end

        function abind(a, b, x, y)
            apos = a + (x - 1) * (oa - 1)
            bpos = b + (y - 1) * (ob - 1)
            return dq1 + bpos + (apos - 1) * bob_ops
        end

        ## q1 × q1ab orthogonality constraints
        for x ∈ 1:ia
            for a1 ∈ 1:oa-1, a2 ∈ a1+1:oa-1, y ∈ 1:ib, b ∈ 1:ob-1
                JuMP.@constraint(model, Γ[aind(a1, x), abind(a2, b, x, y)] == 0)
                JuMP.@constraint(model, Γ[aind(a2, x), abind(a1, b, x, y)] == 0)
            end
        end
        for y ∈ 1:ib
            for b1 ∈ 1:ob-1, b2 ∈ b1+1:ob-1, x ∈ 1:ia, a ∈ 1:oa-1
                JuMP.@constraint(model, Γ[bind(b1, y), abind(a, b2, x, y)] == 0)
                JuMP.@constraint(model, Γ[bind(b2, y), abind(a, b1, x, y)] == 0)
            end
        end

        ## q1 × q1ab self equality constraints
        for x ∈ 1:ia, a ∈ 1:oa-1, y ∈ 1:ib, b ∈ 1:ob-1
            JuMP.@constraint(model, Γ[aind(a, x), abind(a, b, x, y)] == Γ[1, abind(a, b, x, y)])
        end
        for y ∈ 1:ib, b ∈ 1:ob-1, x ∈ 1:ia, a ∈ 1:oa-1
            JuMP.@constraint(model, Γ[bind(b, y), abind(a, b, x, y)] == Γ[1, abind(a, b, x, y)])
        end

        ## q1 × q1ab cross equality constraints
        for x1 ∈ 1:ia, x2 ∈ x1+1:ia
            for a1 ∈ 1:oa-1, a2 ∈ 1:oa-1, y ∈ 1:ib, b ∈ 1:ob-1
                JuMP.@constraint(model, Γ[aind(a1, x1), abind(a2, b, x2, y)] == Γ[aind(a2, x2), abind(a1, b, x1, y)])
            end
        end
        for y1 ∈ 1:ib, y2 ∈ y1+1:ib
            for b1 ∈ 1:ob-1, b2 ∈ 1:ob-1, x ∈ 1:ia, a ∈ 1:oa-1
                JuMP.@constraint(model, Γ[bind(b1, y1), abind(a, b2, x, y2)] == Γ[bind(b2, y2), abind(a, b1, x, y1)])
            end
        end

        ## q1ab × q1ab cross equality constraints
        for x ∈ 1:ia, a ∈ 1:oa-1
            for y1 ∈ 1:ib, y2 ∈ y1+1:ib, b1 ∈ 1:ob-1, b2 ∈ 1:ob-1
                JuMP.@constraint(
                    model,
                    Γ[abind(a, b1, x, y1), abind(a, b2, x, y2)] == Γ[bind(b1, y1), abind(a, b2, x, y2)]
                )
            end
        end
        for y ∈ 1:ib, b ∈ 1:ob-1
            for x1 ∈ 1:ia, x2 ∈ x1+1:ia, a1 ∈ 1:oa-1, a2 ∈ 1:oa-1
                JuMP.@constraint(
                    model,
                    Γ[abind(a1, b, x1, y), abind(a2, b, x2, y)] == Γ[aind(a1, x1), abind(a2, b, x2, y)]
                )
            end
        end

        ## q1ab × q1ab orthogonality constraints
        for x ∈ 1:ia, a1 ∈ 1:oa-1, a2 ∈ a1+1:oa-1
            for y1 ∈ 1:ib, y2 ∈ 1:ib, b1 ∈ 1:ob-1, b2 ∈ 1:ob-1
                JuMP.@constraint(model, Γ[abind(a1, b1, x, y1), abind(a2, b2, x, y2)] == 0)
            end
        end
        for y ∈ 1:ib, b1 ∈ 1:ob-1, b2 ∈ b1+1:ob-1
            for x1 ∈ 1:ia, x2 ∈ 1:ia, a1 ∈ 1:oa-1, a2 ∈ 1:oa-1
                JuMP.@constraint(model, Γ[abind(a1, b1, x1, y), abind(a2, b2, x2, y)] == 0)
            end
        end
    end

    objective = dot(CG, behaviour)
    JuMP.@objective(model, Max, objective)
    if dualize
        JuMP.set_optimizer(model, Dualization.dual_optimizer(solver; coefficient_type = T))
    else
        JuMP.set_optimizer(model, solver)
    end
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model) || @warn JuMP.raw_status(model)
    return JuMP.objective_value(model)::T, JuMP.value.(behaviour)::Matrix{T}
end

function _tsirelson_bound_manual(FC::Matrix{T}, include_ab::Bool; verbose, dualize, solver) where {T<:AbstractFloat}
    ia, ib = size(FC) .- 1
    dq1 = 1 + ia + ib
    dq1ab = dq1 + ia * ib
    model = JuMP.GenericModel{T}()
    dΓ = include_ab ? dq1ab : dq1
    JuMP.@variable(model, Γ[1:dΓ, 1:dΓ] in JuMP.PSDCone())
    ## normalization constraints
    for i ∈ 1:dq1
        JuMP.@constraint(model, Γ[i, i] == 1)
    end

    alice_marginal = Γ[1, 2:ia+1]
    bob_marginal = Γ[1, ia+2:dq1]
    correlation = Γ[2:ia+1, ia+2:dq1]
    behaviour = [Γ[1, 1] bob_marginal'; alice_marginal correlation]

    if include_ab
        ## first line of ab
        JuMP.@constraint(model, vec(correlation') .== Γ[1, dq1+1:dq1ab])

        ## more normalization
        for i ∈ dq1+1:dq1ab
            JuMP.@constraint(model, Γ[i, i] == 1)
        end

        f(x, y) = dq1 + y + (x - 1) * ib
        ## q1 × q1ab self equality constraints
        for x ∈ 1:ia, y ∈ 1:ib
            JuMP.@constraint(model, Γ[1+x, f(x, y)] == Γ[1, 1+ia+y])
        end
        for y ∈ 1:ib, x ∈ 1:ia
            JuMP.@constraint(model, Γ[1+ia+y, f(x, y)] == Γ[1, 1+x])
        end

        ## q1 × q1ab cross equality constraints
        for x1 ∈ 1:ia, x2 ∈ x1+1:ia, y ∈ 1:ib
            JuMP.@constraint(model, Γ[1+x1, f(x2, y)] == Γ[1+x2, f(x1, y)])
        end
        for y1 ∈ 1:ib, y2 ∈ y1+1:ib, x ∈ 1:ia
            JuMP.@constraint(model, Γ[1+ia+y1, f(x, y2)] == Γ[1+ia+y2, f(x, y1)])
        end

        ## q1ab × q1ab cross equality constraints
        for x ∈ 1:ia
            for y1 ∈ 1:ib, y2 ∈ y1+1:ib
                JuMP.@constraint(model, Γ[f(x, y1), f(x, y2)] == Γ[1+ia+y1, 1+ia+y2])
            end
        end
        for y ∈ 1:ib
            for x1 ∈ 1:ia, x2 ∈ x1+1:ia
                JuMP.@constraint(model, Γ[f(x1, y), f(x2, y)] == Γ[1+x1, 1+x2])
            end
        end
    end

    objective = dot(FC, behaviour)
    JuMP.@objective(model, Max, objective)
    if dualize
        JuMP.set_optimizer(model, Dualization.dual_optimizer(solver; coefficient_type = T))
    else
        JuMP.set_optimizer(model, solver)
    end
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model) || @warn JuMP.raw_status(model)
    return JuMP.objective_value(model)::T, JuMP.value.(behaviour)::Matrix{T}
end
