"""
    local_bound(G::Array{T,N}; correlation = N < 4, marg = true)

Computes the local bound of a multipartite Bell functional `G` given as an `N`-dimensional array.
If `correlation` is `false`, `G` is assumed to be written in probability notation.
If `correlation` is `true`, `G` is assumed to be written in correlation notation, with or without marginals depending on `marg`.

Reference: Araújo, Hirsch, Quintino, [arXiv:2005.13418](https://arxiv.org/abs/2005.13418)
"""
function local_bound(G::Array{T,N}; correlation::Bool = N < 4, marg::Bool = true) where {T<:Real,N}
    if correlation
        return _local_bound_correlation(G; marg)
    else
        return _local_bound_probability(G)
    end
end
export local_bound

function _local_bound_correlation(G::Array{T,N}; marg::Bool = true) where {T<:Real,N}
    outs = fill(2, N)
    ins = size(G)

    num_strategies = outs .^ (ins .- 1)
    largest_party = argmax(num_strategies)
    if largest_party != 1
        perm = [largest_party; 2:largest_party-1; 1; largest_party+1:N]
        ins::NTuple{N,Int} = ins[perm]
        G = permutedims(G, perm)
    end

    chunks = _partition(outs[N]^(ins[N] - marg), Threads.nthreads())
    G2 = FArray(G) #workaround for https://github.com/JuliaLang/julia/issues/15276
    tasks = map(chunks) do chunk
        Threads.@spawn _local_bound_correlation_recursive!(copy(G2), chunk, marg)
    end
    score::T = maximum(fetch.(tasks))
    return score
end

_scratch_type(::FArray{T,N}) where {T,N} = N == 2 ? Vector{FVector{T}} : Vector{FArray{T}}

function _local_bound_correlation_recursive!(
    A::FArray{T,N},
    chunk,
    marg = true,
    m = size(A),
    tmp = [fzeros(T, m[1:i]) for i ∈ 1:N-1]::_scratch_type(A),
    offset = [fzeros(T, m[1:i]) for i ∈ 1:N-1]::_scratch_type(A),
    ind = [fzeros(Int8, m[i] - marg) for i ∈ 2:N]
) where {T<:Real,N}
    tmp_end = tmp[N-1]::FArray{T,N - 1}
    offset_end = offset[N-1]::FArray{T,N - 1}
    sum!(offset_end, A)
    A .*= 2
    marg && (offset_end .-= selectdim(A, N, 1)) # note this is twice the original A
    offset_end .*= -1
    chunk[1] > 0 && digits!(ind[N-1], chunk[1] - 1; base = 2)
    score = typemin(T)
    for _ ∈ chunk[1]:chunk[2]
        tmp_end .= offset_end
        _tensor_contraction!(tmp_end, A, ind[N-1], marg)
        temp_score = _local_bound_correlation_recursive!(
            tmp_end,
            (0, 2^(m[N-1] - marg) - 1),
            marg,
            m[1:N-1],
            tmp,
            offset,
            ind
        )
        if temp_score > score
            score = temp_score
        end
        _update_odometer!(ind[N-1], 2)
    end
    return score
end

function _local_bound_correlation_recursive!(A::FVector, chunk, marg, m, tmp, offset, ind)
    score = marg ? A[1] : abs(A[1])
    for x ∈ 2:m[1]
        score += abs(A[x])
    end
    return score
end

function _tensor_contraction!(tmp, A, ind, marg)
    @inbounds for x ∈ eachindex(ind)
        if ind[x] == 1
            for ci ∈ CartesianIndices(tmp)
                tmp[ci] += A[ci, x+marg]
            end
        end
    end
end

function _local_bound_probability(G::Array{T,N2}) where {T<:Real,N2}
    @assert iseven(N2)
    N = N2 ÷ 2
    scenario = size(G)
    outs = scenario[1:N]
    ins = scenario[N+1:2N]

    num_strategies = outs .^ ins
    largest_party = argmax(num_strategies)
    if largest_party != 1
        perm = [largest_party; 2:largest_party-1; 1; largest_party+1:N]
        outs::NTuple{N,Int} = outs[perm]
        ins::NTuple{N,Int} = ins[perm]
        G = permutedims(G, [perm; perm .+ N])
    end
    permutedG = permutedims(G, [1; N + 1; 2:N; N+2:2N])
    squareG = reshape(permutedG, outs[1] * ins[1], prod(outs[2:N]) * prod(ins[2:N]))

    chunks = _partition(prod((outs .^ ins)[2:N]), Threads.nthreads())
    outs2 = outs
    ins2 = ins #workaround for https://github.com/JuliaLang/julia/issues/15276
    tasks = map(chunks) do chunk
        Threads.@spawn _local_bound_probability_core(chunk, outs2, ins2, squareG)
    end
    score::T = maximum(fetch.(tasks))
    return score
end

function _local_bound_probability_core(chunk, outs::NTuple{2,Int}, ins::NTuple{2,Int}, squareG::Array{T,2}) where {T}
    oa, ob = outs
    ia, ib = ins
    score = typemin(T)
    ind = digits(chunk[1] - 1; base = ob, pad = ib)
    offset = Vector(1 .+ ob * (0:ib-1))
    offset_ind = zeros(Int, ib)
    Galice = zeros(T, oa * ia)
    @inbounds for _ ∈ chunk[1]:chunk[2]
        offset_ind .= ind .+ offset
        @views sum!(Galice, squareG[:, offset_ind])
        temp_score = _maxcols!(Galice, oa, ia)
        score = max(score, temp_score)
        _update_odometer!(ind, ob)
    end
    return score
end

function _local_bound_probability_core(chunk, outs::NTuple{N,Int}, ins::NTuple{N,Int}, squareG::Array{T,2}) where {T,N}
    score = typemin(T)
    base = reduce(vcat, [fill(outs[i], ins[i]) for i ∈ 2:N])
    ind = _digits(chunk[1] - 1; base)
    Galice = zeros(T, outs[1] * ins[1])
    sumins = zeros(Int, N - 1)
    for i ∈ 2:N-1
        sumins[i] = sum(ins[2:i])
    end
    sizes = (outs[2:N]..., ins[2:N]...)
    prodsizes = ones(Int, 2N - 2)
    for i ∈ 2:length(prodsizes)
        prodsizes[i] = prod(sizes[1:i-1])
    end
    linearindex(v) = 1 + dot(v, prodsizes)
    by = zeros(Int, 2 * (N - 1))
    ins_region = CartesianIndices(ins[2:N])
    offset_ind = zeros(Int, prod(ins[2:N]))
    @inbounds for _ ∈ chunk[1]:chunk[2]
        counter = 0
        for y ∈ ins_region
            for i ∈ 1:length(y)
                by[i] = ind[y[i]+sumins[i]]
            end
            for i ∈ 1:N-1
                by[i+N-1] = y[i] - 1
            end
            counter += 1
            offset_ind[counter] = linearindex(by)
        end
        @views sum!(Galice, squareG[:, offset_ind])
        temp_score = _maxcols!(Galice, outs[1], ins[1])
        score = max(score, temp_score)
        _update_odometer!(ind, base)
    end
    return score
end

"""
    _maxcols!(A::Array, n::Integer, m::Integer)

Computes `sum(maximum(A, dims = 1))`, with `A` interpreted as an `n` by `m` matrix. `A` is destroyed.
"""
function _maxcols!(v, oa, ia)
    for x ∈ 1:ia
        for a ∈ 2:oa
            if v[a+(x-1)*oa] > v[1+(x-1)*oa]
                v[1+(x-1)*oa] = v[a+(x-1)*oa]
            end
        end
    end
    temp_score = v[1]
    for x ∈ 2:ia
        temp_score += v[1+(x-1)*oa]
    end
    return temp_score
end

"""
    partition(n::Integer, k::Integer)

If `n ≥ k` partitions the set `1:n` into `k` parts as equally sized as possible.
Otherwise partitions it into `n` parts of size 1.
"""
function _partition(n::T, k::T) where {T<:Integer}
    num_parts = min(k, n)
    parts = Vector{Tuple{T,T}}(undef, num_parts)
    base_size = div(n, k)
    num_larger = rem(n, k)
    if num_larger > 0
        parts[1] = (1, base_size + 1)
    else
        parts[1] = (1, base_size)
    end
    i = 2
    while i ≤ num_larger
        parts[i] = (1, base_size + 1) .+ parts[i-1][2]
        i += 1
    end
    while i ≤ num_parts
        parts[i] = (1, base_size) .+ parts[i-1][2]
        i += 1
    end
    return parts
end

function _digits(ind; base)
    N = length(base)
    digits = zeros(Int, N)
    i = 0
    @inbounds while (ind != 0 && i ≤ N - 1)
        i += 1
        ind, digits[i] = divrem(ind, base[i])
    end
    return digits
end

function _update_odometer!(ind::AbstractVector{<:Integer}, base::AbstractVector{<:Integer})
    ind[1] += 1
    d = length(ind)

    @inbounds for i ∈ 1:d
        if ind[i] ≥ base[i]
            ind[i] = 0
            i < d ? ind[i+1] += 1 : return
        else
            return
        end
    end
end

function _update_odometer!(ind::AbstractVector{<:Integer}, base::Integer)
    ind[1] += 1
    d = length(ind)

    @inbounds for i ∈ 1:d
        if ind[i] ≥ base
            ind[i] = 0
            i < d ? ind[i+1] += 1 : return
        else
            return
        end
    end
end

"""
    tensor_collinsgisin(p::Array, behaviour::Bool = false; correlation::Bool = false)

Converts a multipartite Bell functional `p` into Collins-Gisin notation.
If `correlation` is `true`, `p` is assumed to be written in correlation notation. Otherwise, `p` is assumed to be written in probability notation.
If `behaviour` is `true` do instead the transformation for behaviours. Doesn't assume normalization.

Also accepts the arguments of `tensor_probability` (state and measurements) for convenience.
"""
function tensor_collinsgisin(p::AbstractArray, behaviour::Bool = false; correlation::Bool = false)
    if correlation
        return _tensor_collinsgisin_correlation(p, behaviour)
    else
        return _tensor_collinsgisin_probability(p, behaviour)
    end
end
# accepts directly the arguments of tensor_probability
function tensor_collinsgisin(rho::Hermitian, all_Aax::Vector{<:Measurement}...)
    return tensor_collinsgisin(tensor_probability(rho, all_Aax...), true)
end
# shorthand syntax for identical measurements on all parties
function tensor_collinsgisin(rho::Hermitian, Aax::Vector{<:Measurement}, N::Integer)
    return tensor_collinsgisin(rho, fill(Aax, N)...)
end
export tensor_collinsgisin

function _tensor_collinsgisin_probability(p::AbstractArray{T,N2}, behaviour::Bool = false) where {T,N2}
    @assert iseven(N2)
    N = N2 ÷ 2
    scenario = size(p)
    outs = scenario[1:N]
    ins = scenario[N+1:2N]
    cgindex(a, x) = (a .!= outs) .* (a .+ (x .- 1) .* (outs .- 1)) .+ 1
    CG = zeros(_solver_type(T), ins .* (outs .- 1) .+ 1)

    if !behaviour
        for x ∈ CartesianIndices(ins)
            for a ∈ CartesianIndices(outs)
                for a2 ∈ Iterators.product(union.(a.I, outs)...)
                    signdiff = isodd(sum(a.I .!= outs) - sum(a2 .!= outs))
                    CG[cgindex(a.I, x.I)...] += (-1)^signdiff * p[a2..., x]
                end
            end
        end
    else
        for x ∈ CartesianIndices(ins)
            for a ∈ CartesianIndices(outs)
                normalization = T(1) / _normalization_tensor(outs, ins, a.I)
                cgiterators = map((i, j) -> i == j ? (1:j) : (i:i), a.I, outs)
                for a2 ∈ CartesianIndices(cgiterators)
                    CG[cgindex(a.I, x.I)...] += p[a2, x] * normalization
                end
            end
        end
    end
    return CG
end

function _tensor_collinsgisin_correlation(FC::AbstractArray{T}, behaviour::Bool = false) where {T}
    dims = size(FC)
    CG = zeros(_solver_type(T), dims)
    if !behaviour
        for x ∈ CartesianIndices(dims)
            n = sum(x.I .!= 1)
            for x2 ∈ Iterators.product(union.(x.I, 1)...)
                s = sum(x2 .!= 1)
                CG[x2...] += (-1)^isodd(n - s) * 2^s * FC[x]
            end
        end
    else
        for x ∈ CartesianIndices(dims)
            n = sum(x.I .!= 1)
            for x2 ∈ Iterators.product(union.(x.I, 1)...)
                CG[x] += FC[x2...] / 2^n
            end
        end
    end
    return CG
end

function _normalization_tensor(outs::NTuple{N,Int}, ins::NTuple{N,Int}, var::NTuple{N,Int}) where {N}
    normalization = 1
    @inbounds for i ∈ 1:N
        if var[i] == outs[i]
            normalization *= ins[i]
        end
    end
    return normalization
end

"""
    tensor_probability(CG::Array, scenario::Tuple, behaviour::Bool = false)

Takes a multipartite Bell functional `CG` in Collins-Gisin notation and transforms it to probability notation.
`scenario` is a tuple detailing the number of inputs and outputs, in the order (oa, ob, ..., ia, ib, ...).
If `behaviour` is `true` do instead the transformation for behaviours. Doesn't assume normalization.
"""
function tensor_probability(CG::AbstractArray{T,N}, scenario::Tuple, behaviour::Bool = false) where {T,N}
    p = zeros(_solver_type(T), scenario)
    outs = scenario[1:N]
    ins = scenario[N+1:2N]
    cgindex(a, x) = (a .!= outs) .* (a .+ (x .- 1) .* (outs .- 1)) .+ 1

    if !behaviour
        for x ∈ CartesianIndices(ins)
            for a ∈ CartesianIndices(outs)
                for a2 ∈ Iterators.product(union.(a.I, outs)...)
                    p[a, x] += CG[cgindex(a2, x.I)...] / _normalization_tensor(outs, ins, a2)
                end
            end
        end
    else
        for x ∈ CartesianIndices(ins)
            for a ∈ CartesianIndices(outs)
                cgiterators = map((i, j) -> i == j ? (1:j) : (i:i), a.I, outs)
                for a2 ∈ CartesianIndices(cgiterators)
                    signdiff = isodd(sum(a.I .!= outs) - sum(a2.I .!= outs))
                    p[a, x] += (-1)^signdiff * CG[cgindex(a2.I, x.I)...]
                end
            end
        end
    end
    return p
end

"""
    tensor_probability(FC::Matrix, behaviour::Bool = false)

Takes a bipartite Bell functional `FC` in correlation notation and transforms it to probability notation.
If `behaviour` is `true` do instead the transformation for behaviours. Doesn't assume normalization.
"""
function tensor_probability(FC::AbstractArray{T,N}, behaviour::Bool = false) where {T,N}
    o = ntuple(i -> 2, N)
    m = size(FC) .- 1
    FP = zeros(T, o..., m...)
    cia = CartesianIndices(o)
    cix = CartesianIndices(m)
    # there may be a smarter way to order these loops
    for a2 ∈ cia
        ind = collect(a2.I) .== 2
        denominator = behaviour ? 1 : prod(m[.!ind]; init = 1)
        for a1 ∈ cia
            s = (-1)^(sum(a1.I[ind]; init = 0) - sum(ind))
            for x ∈ cix
                FP[a1, x] += s * FC[[a2[n] == 1 ? 1 : x[n] + 1 for n ∈ 1:N]...] / denominator
            end
        end
    end
    if behaviour
        FP ./= 2^N
    end
    cleanup!(FP)
    return FP
end

"""
    tensor_probability(rho::Hermitian, all_Aax::Vector{Measurement}...)
    tensor_probability(rho::Hermitian, Aax::Vector{Measurement}, N::Integer)

Applies N sets of measurements onto a state `rho` to form a probability array.
If all parties apply the same measurements, use the shorthand notation.
"""
function tensor_probability(
    rho::Hermitian{T1,Matrix{T1}},
    first_Aax::Vector{Measurement{T2}}, # needed so that T2 is not unbounded
    other_Aax::Vector{Measurement{T2}}...
) where {T1,T2}
    T = real(promote_type(T1, T2))
    all_Aax = (first_Aax, other_Aax...)
    N = length(all_Aax)
    m = length.(all_Aax) # numbers of inputs per party
    o = broadcast(Aax -> maximum(length.(Aax)), all_Aax) # numbers of outputs per party
    FP = zeros(T, o..., m...)
    cia = CartesianIndices(o)
    cix = CartesianIndices(m)
    for a ∈ cia, x ∈ cix
        if all([a[n] ≤ length(all_Aax[n][x[n]]) for n ∈ 1:N])
            FP[a, x] = real(dot(Hermitian(kron([all_Aax[n][x[n]][a[n]] for n ∈ 1:N]...)), rho))
        end
    end
    return FP
end
# shorthand syntax for identical measurements on all parties
function tensor_probability(rho::Hermitian, Aax::Vector{<:Measurement}, N::Integer)
    return tensor_probability(rho, fill(Aax, N)...)
end
export tensor_probability

"""
    tensor_correlation(p::AbstractArray, behaviour::Bool = false; collinsgisin::Bool = false, marg::Bool = true)

Converts a multipartite Bell functional `p` into correlation notation.
If `collinsgisin` is `true`, `p` is assumed to be written in Collins-Gisin notation. Otherwise, `p` is assumed to be written in probability notation.
If `marg` is `false`, the output contains only full correlators, with no marginals.
If `behaviour` is `true` do the transformation for behaviours. Doesn't assume normalization.

Also accepts the arguments of `tensor_probability` (state and measurements) for convenience.
"""
function tensor_correlation(p::AbstractArray, behaviour::Bool = false; collinsgisin::Bool = false, marg::Bool = true)
    if collinsgisin
        return _tensor_correlation_collinsgisin(p, behaviour; marg)
    else
        return _tensor_correlation_probability(p, behaviour; marg)
    end
end
# accepts directly the arguments of tensor_probability
# avoids creating the full probability tensor for performance
function tensor_correlation(
    rho::Hermitian{T1,Matrix{T1}},
    first_Aax::Vector{Measurement{T2}}, # needed so that T2 is not unbounded
    other_Aax::Vector{Measurement{T2}}...;
    marg::Bool = true
) where {T1,T2}
    T = real(promote_type(T1, T2))
    all_Aax = (first_Aax, other_Aax...)
    N = length(all_Aax)
    m = Tuple(length.(all_Aax)) # numbers of inputs per party
    o = Tuple(broadcast(Aax -> maximum(length.(Aax)), all_Aax)) # numbers of outputs per party
    @assert all(o .== 2)
    @assert all(broadcast(Aax -> minimum(length.(Aax)), all_Aax) .== 2) # sanity check
    size_FC = marg ? m .+ 1 : m
    FC = zeros(T, size_FC)
    cia = CartesianIndices(o)
    cix = CartesianIndices(size_FC)
    for a ∈ cia, x ∈ cix
        obs = [x[n] > marg ? all_Aax[n][x[n]-marg][1] - all_Aax[n][x[n]-marg][2] : one(all_Aax[n][1][1]) for n ∈ 1:N]
        FC[x] = real(dot(Hermitian(kron(obs...)), rho))
    end
    return FC
end
# shorthand syntax for identical measurements on all parties
function tensor_correlation(rho::Hermitian, Aax::Vector{<:Measurement}, N::Integer; marg::Bool = true)
    return tensor_correlation(rho, fill(Aax, N)...; marg)
end
export tensor_correlation

function _tensor_correlation_probability(
    p::AbstractArray{T,N2},
    behaviour::Bool = false;
    marg::Bool = true
) where {T,N2}
    @assert iseven(N2)
    N = N2 ÷ 2
    o = size(p)[1:N] # numbers of outputs per party
    @assert all(o .== 2)
    m = size(p)[N+1:end] # numbers of inputs per party
    size_FC = marg ? m .+ 1 : m
    FC = zeros(_solver_type(T), size_FC)
    cia = CartesianIndices(o)
    cix = CartesianIndices(size_FC)
    for x ∈ cix
        # separating here prevent the need of the iterate function on unique elements of type T
        if all(x.I .> marg)
            FC[x] = sum((-1)^sum(a[n] - 1 for n ∈ 1:N if x[n] > marg; init = 0) * p[a, (x.I .- marg)...] for a ∈ cia)
        else
            x_colon = Union{Colon,Int}[x[n] > marg ? x[n] - marg : Colon() for n ∈ 1:N]
            FC[x] = sum((-1)^sum(a[n] - 1 for n ∈ 1:N if x[n] > marg; init = 0) * sum(p[a, x_colon...]) for a ∈ cia)
        end
    end
    if !behaviour
        FC ./= 2^N
    elseif marg
        for n ∈ 1:N
            x_colon = Union{Colon,Int}[i == n ? 1 : Colon() for i ∈ 1:N]
            FC[x_colon...] ./= m[n]
        end
    end
    cleanup!(FC)
    return FC
end

function _tensor_correlation_collinsgisin(
    CG::AbstractArray{T,N},
    behaviour::Bool = false;
    marg::Bool = true
) where {T,N}
    m = size(CG) .- 1
    size_FC = marg ? m .+ 1 : m
    FC = zeros(_solver_type(T), size_FC)
    if !behaviour
        if !marg
            for x ∈ CartesianIndices(size_FC)
                FC[x] = CG[(x.I .+ 1)...] / 2^N
            end
        else
            for x ∈ CartesianIndices(size_FC)
                n = sum(x.I .!= 1)
                for x2 ∈ Iterators.product(union.(x.I, 1)...)
                    FC[x2...] += CG[x] / 2^n
                end
            end
        end
    else
        for x ∈ CartesianIndices(size_FC)
            n = marg ? sum(x.I .!= 1) : N
            for x2 ∈ Iterators.product(union.(x.I .+ !marg, 1)...)
                s = sum(x2 .!= 1)
                FC[x] += (-1)^isodd(n - s) * 2^s * CG[x2...]
            end
        end
    end
    return FC
end

"""
    nonlocality_robustness(FP::Array; noise::String = "white", verbose::Bool = false, solver = Hypatia.Optimizer{_solver_type(T)})

Computes the nonlocality robustness of the behaviour `FP`. Argument `noise` indicates the kind of noise to be used: "white" (default), "local", or "general".

Reference: Baek, Ryu, Lee, [arxiv:2311.07077](https://arxiv.org/abs/2311.07077)
"""
function nonlocality_robustness(
    FP::Array{T,N2};
    noise::String = "white",
    verbose = false,
    solver = Hypatia.Optimizer{_solver_type(T)}
) where {T<:Real,N2}
    @assert noise ∈ ["white", "local", "general"]

    @assert iseven(N2)
    N = N2 ÷ 2
    scenario = size(FP)
    temp_outs::NTuple{N,Int} = scenario[1:N]
    temp_ins::NTuple{N,Int} = scenario[N+1:2N]
    num_strategies::NTuple{N,Int} = temp_outs .^ temp_ins

    normalization = sum(FP[1:prod(temp_outs)])

    exploding_party = findfirst(x -> x == 0, num_strategies)
    largest_party = exploding_party == nothing ? argmax(num_strategies) : exploding_party
    if largest_party != 1
        perm = [largest_party; 2:largest_party-1; 1; largest_party+1:N]
        temp_outs = temp_outs[perm]
        temp_ins = temp_ins[perm]
        num_strategies = num_strategies[perm]
        FP = permutedims(FP, [perm; perm .+ N])
    end
    outs, ins = temp_outs, temp_ins #workaround for https://github.com/JuliaLang/julia/issues/15276
    total_num_strategies = prod(num_strategies[2:N])

    stT = _solver_type(T)
    model = JuMP.GenericModel{stT}()

    JuMP.@variable(model, t)
    JuMP.@variable(model, π[1:total_num_strategies])
    p = [JuMP.@variable(model, [1:outs[1]-1, 1:ins[1]], lower_bound = 0) for _ ∈ 1:total_num_strategies]

    last_p = [JuMP.@expression(model, π[λ] - sum(p[λ][:, x])) for λ ∈ 1:total_num_strategies, x ∈ 1:ins[1]]
    jumpT = typeof(1 * t)
    local_model = Array{jumpT,N2}(undef, size(FP))
    for i ∈ eachindex(local_model)
        local_model[i] = 0
    end

    q = Vector{Matrix{typeof(t)}}(undef, 0)
    if noise == "local"
        JuMP.@variable(model, ξ[1:total_num_strategies])
        resize!(q, total_num_strategies)
        for i ∈ eachindex(q)
            q[i] = JuMP.@variable(model, [1:outs[1]-1, 1:ins[1]], lower_bound = 0)
        end

        last_q = [JuMP.@expression(model, ξ[λ] - sum(q[λ][:, x])) for λ ∈ 1:total_num_strategies, x ∈ 1:ins[1]]
        local_noise = Array{jumpT,N2}(undef, size(FP))
        for i ∈ eachindex(local_noise)
            local_noise[i] = 0
        end
    end

    base = reduce(vcat, [fill(outs[i], ins[i]) for i ∈ 2:N])
    strategy = Vector{Int}(undef, sum(ins[2:N]))
    b = Vector{Int}(undef, N - 1)
    for λ ∈ 1:total_num_strategies
        strategy = _digits(λ - 1; base)
        for y ∈ CartesianIndices(ins[2:N])
            shift = 0
            for i ∈ 2:N
                b[i-1] = strategy[shift+y.I[i-1]] + 1
                shift += ins[i]
            end
            for x ∈ 1:ins[1]
                for a ∈ 1:outs[1]-1
                    JuMP.add_to_expression!(local_model[a, b..., x, y], p[λ][a, x])
                    noise == "local" && JuMP.add_to_expression!(local_noise[a, b..., x, y], q[λ][a, x])
                end
                JuMP.add_to_expression!(local_model[outs[1], b..., x, y], last_p[λ, x])
                noise == "local" && JuMP.add_to_expression!(local_noise[outs[1], b..., x, y], last_q[λ, x])
            end
        end
    end

    if noise == "white"
        JuMP.@constraint(model, FP .+ t * normalization / prod(outs) == local_model)
    else
        if noise == "local"
            JuMP.@constraint(model, FP + local_noise == local_model)
            JuMP.@constraint(model, last_q .≥ 0)
        elseif noise == "general"
            JuMP.@constraint(model, FP - local_model .≤ 0)
        end
        JuMP.@constraint(model, sum(π) == (1 + t) * normalization)
    end
    JuMP.@constraint(model, last_p .≥ 0)
    JuMP.@constraint(model, t ≥ 0)

    JuMP.@objective(model, Min, t)

    JuMP.set_optimizer(model, solver)
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model) || error(JuMP.raw_status(model))
    return JuMP.objective_value(model)
end
export nonlocality_robustness
