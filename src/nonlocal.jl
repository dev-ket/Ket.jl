"""
    local_bound(G::Array{T,N}; correlation = N < 4, marg = true)

Computes the local bound of a multipartite Bell functional `G` given as an `N`-dimensional array.
If `correlation` is `false`, `G` is assumed to be written in full probability notation.
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
    G2 = G #workaround for https://github.com/JuliaLang/julia/issues/15276
    tasks = map(chunks) do chunk
        Threads.@spawn _local_bound_correlation_recursive!(copy(G2), chunk, marg)
    end
    score::T = maximum(fetch.(tasks))
    return score
end

function _local_bound_correlation_recursive!(
    A::Array{T,N},
    chunk,
    marg = true,
    m = size(A),
    tmp = [zeros(T, m[1:i]) for i ∈ 1:N-1],
    offset = [zeros(T, m[1:i]) for i ∈ 1:N-1],
    ind = [zeros(Int8, m[i] - marg) for i ∈ 2:N],
) where {T<:Real,N}
    tmp_end::Array{T,N-1} = tmp[N-1]
    offset_end::Array{T,N-1} = offset[N-1]
    sum!(offset_end, A)
    A .*= 2
    marg && (offset_end .-= selectdim(A, N, 1)) # note this is twice the original A
    offset_end .*= -1
    chunk[1] > 0 && digits!(ind[N-1], chunk[1] - 1; base = 2)
    score = typemin(T)
    for _ ∈ chunk[1]:chunk[2]
        tmp_end .= offset_end
        _tensor_contraction!(tmp_end, A, ind[N-1], marg)
        @views temp_score = _local_bound_correlation_recursive!(tmp_end, (0, 2^(m[N-1]-marg)-1), marg, m[1:N-1], tmp[1:N-2], offset[1:N-2], ind[1:N-2])
        if temp_score > score
            score = temp_score
        end
        _update_odometer!(ind[N-1], 2)
    end
    return score
end

function _local_bound_correlation_recursive!(A::Vector, chunk, marg, m, tmp, offset, ind)
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
    @inbounds for i ∈ 1:N
        digits[i] = ind % base[i]
        ind = ind ÷ base[i]
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
    tensor_collinsgisin(p::Array, behaviour::Bool = false)

Takes a multipartite Bell functional `p` in full probability notation and transforms it to Collins-Gisin notation.
If `behaviour` is `true` do instead the transformation for behaviours. Doesn't assume normalization.

Also accepts the arguments of `tensor_probability` (state and measurements) for convenience.
"""
function tensor_collinsgisin(p::AbstractArray{T,N2}, behaviour::Bool = false) where {T,N2}
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
                    ndiff = abs(sum(a.I .!= outs) - sum(a2 .!= outs))
                    CG[cgindex(a.I, x.I)...] += (-1)^ndiff * p[a2..., x]
                end
            end
        end
    else
        for x ∈ CartesianIndices(ins)
            for a ∈ CartesianIndices(outs)
                cgiterators = map((i, j) -> i == j ? (1:j) : (i:i), a.I, outs)
                for a2 ∈ CartesianIndices(cgiterators)
                    CG[cgindex(a.I, x.I)...] += p[a2, x] / prod(ins[BitVector(a.I .== outs)])
                end
            end
        end
    end
    return CG
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

"""
    tensor_probability(CG::Array, scenario::AbstractVecOrTuple, behaviour::Bool = false)

Takes a multipartite Bell functional `CG` in Collins-Gisin notation and transforms it to full probability notation.
`scenario` is a tuple detailing the number of inputs and outputs, in the order (oa, ob, ..., ia, ib, ...).
If `behaviour` is `true` do instead the transformation for behaviours. Doesn't assume normalization.
"""
function tensor_probability(
    CG::AbstractArray{T,N},
    scenario::AbstractVecOrTuple{<:Integer},
    behaviour::Bool = false
) where {T,N}
    p = zeros(_solver_type(T), scenario...)
    outs = Tuple(scenario[1:N])
    ins = Tuple(scenario[N+1:2N])
    cgindex(a, x) = (a .!= outs) .* (a .+ (x .- 1) .* (outs .- 1)) .+ 1

    if !behaviour
        for x ∈ CartesianIndices(ins)
            for a ∈ CartesianIndices(outs)
                for a2 ∈ Iterators.product(union.(a.I, outs)...)
                    p[a, x] += CG[cgindex(a2, x.I)...] / prod(ins[BitVector(a2 .== outs)])
                end
            end
        end
    else
        for x ∈ CartesianIndices(ins)
            for a ∈ CartesianIndices(outs)
                cgiterators = map((i, j) -> i == j ? (1:j) : (i:i), a.I, outs)
                for a2 ∈ CartesianIndices(cgiterators)
                    ndiff = abs(sum(a.I .!= outs) - sum(a2.I .!= outs))
                    p[a, x] += (-1)^ndiff * CG[cgindex(a2.I, x.I)...]
                end
            end
        end
    end
    return p
end

"""
    tensor_probability(FC::Matrix, behaviour::Bool = false)

Takes a bipartite Bell functional `FC` in full correlator notation and transforms it to full probability notation.
If `behaviour` is `true` do instead the transformation for behaviours. Doesn't assume normalization.
"""
function tensor_probability(FC::AbstractArray{T,N}, behaviour::Bool = false) where {T,N}
    o = Tuple(fill(2, N))
    m = size(FC) .- 1
    FP = zeros(T, o..., m...)
    cia = CartesianIndices(o)
    cix = CartesianIndices(m)
    # there may be a smarter way to order these loops
    for a2 ∈ cia
        ind = collect(a2.I) .== 2
        denominator = behaviour ? 1 : prod(m[.!ind]; init = 1)
        for a1 ∈ cia
            s = (-1)^sum(a1.I[ind] .- 1; init = 0)
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
    tensor_correlation(p::AbstractArray{T, N2}, behaviour::Bool = false; marg::Bool = true)

Converts a 2 × … × 2 × m × … × m probability array into
- an m × … × m correlation array (no marginals)
- an (m+1) × … × (m+1) correlation array (marginals).
If `behaviour` is `true` do the transformation for behaviours. Doesn't assume normalization.

Also accepts the arguments of `tensor_probability` (state and measurements) for convenience.
"""
function tensor_correlation(p::AbstractArray{T,N2}, behaviour::Bool = false; marg::Bool = true) where {T,N2}
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
