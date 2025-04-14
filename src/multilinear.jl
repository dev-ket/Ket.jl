"""
    _tidx(idx::Integer, dims::Vector)

Converts a standard index `idx` to a tensor index [i₁, i₂, ...] with subsystems dimensions `dims`.
"""
function _tidx(idx::Integer, dims::Vector{<:Integer})
    result = Vector{Int}(undef, length(dims))
    _tidx!(result, idx, dims)
    return result
end

function _tidx!(tidx::AbstractVector{<:Integer}, idx::Integer, dims::Vector{<:Integer})
    nsys = length(dims)
    cidx = idx - 1 # Current index
    dr = prod(dims)
    for k ∈ 1:nsys
        # Everytime you increase a tensor index you shift by the product of remaining dimensions
        dr ÷= dims[k]
        tidx[k] = (cidx ÷ dr) + 1
        cidx %= dr
    end
    return tidx
end

"""
    _idx(tidx::Vector, dims::Vector)

Converts a tensor index `tidx` = [i₁, i₂, ...] with subsystems dimensions `dims` to a standard index.
"""
function _idx(tidx::Vector{<:Integer}, dims::Vector{<:Integer})
    i = 1
    shift = 1
    for k ∈ length(tidx):-1:1
        i += (tidx[k] - 1) * shift
        shift *= dims[k]
    end
    return i
end

"""
    _inv_ssys(ssys::AbstractVector, nsys::Integer)

Return the complement of the set of subsystems given ; {x ∈ [1,nsys] : x ∉ ssys}
"""
function _inv_ssys(ssys::AbstractVector{<:Integer}, nsys::Integer)
    inv_ssys = Vector{Integer}(undef, nsys - length(ssys))
    _inv_ssys!(inv_ssys, ssys, nsys)
    return inv_ssys
end

function _inv_ssys!(inv_ssys::AbstractVector{<:Integer}, ssys::AbstractVector{<:Integer}, nsys::Integer)
    isempty(ssys) && return 1:nsys
    nsys_og = length(ssys)
    sorted_ssys = sort(ssys)
    i, j = 1, 1
    for k ∈ 1:nsys
        if j <= nsys_og && k == sorted_ssys[j]
            j += 1
            continue
        end
        inv_ssys[i] = k
        i += 1
    end
    return inv_ssys
end

"""
    _step_sizes_ssys(dims::AbstractVector)

Return the array step_sizes s.t. 
step_sizes[j] is the step in standard index to go from tensor index 
[i₁, i₂, ..., iⱼ, ...] to tensor index [i₁, i₂, ..., iⱼ + 1, ...]
"""
function _step_sizes_ssys(dims::AbstractVector{<:Integer})
    isempty(dims) && return Int[]
    step_sizes = similar(dims)
    return _step_sizes_ssys!(step_sizes, dims)
end

function _step_sizes_ssys!(step_sizes::AbstractVector{<:Integer}, dims::AbstractVector{<:Integer})
    step_sizes[end] = 1
    for i ∈ length(dims)-1:-1:1
        step_sizes[i] = step_sizes[i+1] * dims[i+1]
    end
    return step_sizes
end

"""
    _step_iterator(dims::Vector, step_sizes::Vector)

length(Dims) nested loops of range dims[i] each.
Returns array step_iterator s.t. 
The value at tensor index [a₁, a₂, ...] is 1 + ∑ (aᵢ - 1) * step_sizes[i]
"""
function _step_iterator(dims::Vector{<:Integer}, step_sizes::Vector{<:Integer})
    isempty(dims) && return Int[]
    step_iterator = Vector{Integer}(undef, prod(dims))
    return _step_iterator!(step_iterator, dims, step_sizes)
end

function _step_iterator!(
    step_iterator::AbstractVector{<:Integer},
    dims::AbstractVector{<:Integer},
    step_sizes::AbstractVector{<:Integer}
)
    step_sizes_idx = _step_sizes_ssys(dims)
    _step_iterator_rec!(step_iterator, dims, step_sizes_idx, step_sizes, 1, 1, 1)
    return step_iterator
end

# Helper for _step_iterator
function _step_iterator_rec!(
    res::AbstractVector{<:Integer},
    dims::AbstractVector{<:Integer},
    step_sizes_idx::AbstractVector{<:Integer},
    step_sizes_res::AbstractVector{<:Integer},
    idx::Integer,
    acc::Integer,
    it::Integer
)

    #Base case
    if it == length(dims)
        step_idx = step_sizes_idx[end]
        step_res = step_sizes_res[end]
        res[idx] = acc
        for _ ∈ 2:dims[end] #skip first
            idx += step_idx
            acc += step_res
            res[idx] = acc
        end
        return
    end

    #Rec case
    step_idx = step_sizes_idx[it]
    step_res = step_sizes_res[it]
    _step_iterator_rec!(res, dims, step_sizes_idx, step_sizes_res, idx, acc, it + 1)
    for _ ∈ 2:dims[it] #skip first
        idx += step_idx
        acc += step_res
        _step_iterator_rec!(res, dims, step_sizes_idx, step_sizes_res, idx, acc, it + 1)
    end
end

@doc """
    partial_trace(X::AbstractMatrix, remove::AbstractVector, dims::AbstractVector = _equal_sizes(X))

Takes the partial trace of matrix `X` with subsystem dimensions `dims` over the subsystems in `remove`.
If the argument `dims` is omitted two equally-sized subsystems are assumed.
""" partial_trace(X::AbstractMatrix, remove::AbstractVector, dims::AbstractVector = _equal_sizes(X))
for (T, wrapper) ∈ [(:AbstractMatrix, :identity), (:(Hermitian), :(Hermitian)), (:(Symmetric), :(Symmetric))]
    @eval begin
        function partial_trace(
            X::$T,
            remove::AbstractVector{<:Integer},
            dims::AbstractVector{<:Integer} = _equal_sizes(X)
        )
            isempty(remove) && return X
            length(remove) == length(dims) && return $wrapper([eltype(X)(tr(X));;])

            nsys = length(dims)

            keep = _inv_ssys(remove, nsys)
            ssys_step = _step_sizes_ssys(dims)

            dims_keep = dims[keep] # The tensor dimensions of Y
            dims_rm = dims[remove] # The tensor dimensions of the traced out systems

            dY = prod(dims_keep)    # Dimension of Y
            Y = zeros(eltype(X), (dY, dY))  # Final output Y

            ssys_step_keep = ssys_step[keep]
            ssys_step_rm = ssys_step[remove]

            step_iterator_keep = _step_iterator(dims_keep, ssys_step_keep)
            step_iterator_rm = _step_iterator(dims_rm, ssys_step_rm)
            step_iterator_rm .-= 1

            for k ∈ step_iterator_rm
                view_k_idx = k .+ step_iterator_keep
                X_ssys = @view X[view_k_idx, view_k_idx]
                Y += X_ssys
            end
            if !isbits(Y[1]) #this is a workaround for a bug in Julia ≤ 1.10
                if $T == Hermitian
                    LinearAlgebra.copytri!(Y, 'U', true)
                elseif $T == Symmetric
                    LinearAlgebra.copytri!(Y, 'U')
                end
            end
            return $wrapper(Y)
        end
    end
end
export partial_trace

"""
    partial_trace(X::AbstractMatrix, remove::Integer, dims::AbstractVector = _equal_sizes(X)))

Takes the partial trace of matrix `X` with subsystem dimensions `dims` over the subsystem `remove`.
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
partial_trace(X::AbstractMatrix, remove::Integer, dims::AbstractVector{<:Integer} = _equal_sizes(X)) =
    partial_trace(X, [remove], dims)

@doc """
    partial_transpose(X::AbstractMatrix, transp::AbstractVector, dims::AbstractVector = _equal_sizes(X))

Takes the partial transpose of matrix `X` with subsystem dimensions `dims` on the subsystems in `transp`.
If the argument `dims` is omitted two equally-sized subsystems are assumed.
""" partial_transpose(X::AbstractMatrix, transp::AbstractVector, dims::AbstractVector = _equal_sizes(X))

for (T, wrapper) ∈ [(:AbstractMatrix, :identity), (:(Hermitian), :(Hermitian)), (:(Symmetric), :(Symmetric))]
    @eval begin
        function partial_transpose(
            X::$T,
            transp::AbstractVector{<:Integer},
            dims::AbstractVector{<:Integer} = _equal_sizes(X)
        )
            isempty(transp) && return X
            length(transp) == length(dims) && return $wrapper(collect(transpose(X)))

            keep = Vector{eltype(transp)}(undef, length(dims) - length(transp)) # Systems kept
            counter = 0
            for i ∈ 1:length(dims)
                if i ∉ transp
                    counter += 1
                    keep[counter] = i
                end
            end

            d = size(X, 1)                            # Dimension of the final output Y
            Y = similar(X, (d, d))                    # Final output Y

            tXi = Vector{Int}(undef, length(dims))    # Tensor indexing of X for row
            tXj = Vector{Int}(undef, length(dims))    # Tensor indexing of X for column

            tYi = Vector{Int}(undef, length(dims))    # Tensor indexing of Y for row
            tYj = Vector{Int}(undef, length(dims))    # Tensor indexing of Y for column

            @inbounds for j ∈ 1:d
                _tidx!(tYj, j, dims)
                for i ∈ 1:j-1
                    _tidx!(tYi, i, dims)

                    for k ∈ keep
                        tXi[k] = tYi[k]
                        tXj[k] = tYj[k]
                    end

                    for t ∈ transp
                        tXi[t] = tYj[t]
                        tXj[t] = tYi[t]
                    end

                    Xi, Xj = _idx(tXi, dims), _idx(tXj, dims)
                    Y[i, j] = X[Xi, Xj]
                    Y[j, i] = X[Xj, Xi]
                end
                for k ∈ keep
                    tXj[k] = tYj[k]
                end

                for t ∈ transp
                    tXj[t] = tYj[t]
                end

                Xj = _idx(tXj, dims)
                Y[j, j] = X[Xj, Xj]
            end
            return $wrapper(Y)
        end
    end
end
export partial_transpose

"""
    partial_transpose(X::AbstractMatrix, transp::Integer, dims::AbstractVector = _equal_sizes(X))

Takes the partial transpose of matrix `X` with subsystem dimensions `dims` on the subsystem `transp`.
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
partial_transpose(X::AbstractMatrix, transp::Integer, dims::AbstractVector{<:Integer} = _equal_sizes(X)) =
    partial_transpose(X, [transp], dims)

"""
    _idxperm(perm::Vector, dims::Vector)

Computes the index permutation associated with permuting the subsystems of a vector with subsystem dimensions `dims` according to `perm`.
"""
function _idxperm(perm::Vector{<:Integer}, dims::Vector{<:Integer})
    p = Vector{Int}(undef, prod(dims))
    _idxperm!(p, perm, dims)
    return p
end

function _idxperm!(p::Vector{<:Integer}, perm::Vector{<:Integer}, dims::Vector{<:Integer})
    subsystem_og_step = _step_sizes_ssys(dims)

    subsystem_perm_step = similar(dims)
    subsystem_perm_step_view = @view subsystem_perm_step[perm]
    dims_view = @view dims[perm]
    _step_sizes_ssys!(subsystem_perm_step_view, dims_view)
    _step_iterator_rec!(p, dims, subsystem_perm_step, subsystem_og_step, 1, 1, 1)
end

"""
    permute_systems(X::AbstractVector, perm::AbstractVector, dims::AbstractVector = _equal_sizes(X))

Permutes the order of the subsystems of vector `X` with subsystem dimensions `dims` according to the permutation `perm`.
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
function permute_systems(
    X::AbstractVector{T},
    perm::AbstractVector{<:Integer},
    dims::AbstractVector{<:Integer} = _equal_sizes(X)
) where {T}
    perm == 1:length(perm) && return X
    X == 1:length(X) && return _idxperm!(X, perm, dims)
    p = _idxperm(perm, dims)
    return X[p]
end
export permute_systems

@doc """
    permute_systems(X::AbstractMatrix, perm::AbstractVector, dims::AbstractVector = _equal_sizes(X))

Permutes the order of the subsystems of the square matrix `X`, which is composed by square subsystems of dimensions `dims`, according to the permutation `perm`.
If the argument `dims` is omitted two equally-sized subsystems are assumed.
""" permute_systems(
    X::AbstractMatrix,
    perm::AbstractVector,
    dims::AbstractVector = _equal_sizes(X);
    rows_only::Bool = false
)
for (T, wrapper) ∈ [(:AbstractMatrix, :identity), (:(Hermitian), :(Hermitian)), (:(Symmetric), :(Symmetric))]
    @eval begin
        function permute_systems(
            X::$T,
            perm::AbstractVector{<:Integer},
            dims::AbstractVector{<:Integer} = _equal_sizes(X);
            rows_only::Bool = false
        )
            perm == 1:length(perm) && return X
            p = _idxperm(perm, dims)
            return rows_only ? $wrapper(X[p, :]) : $wrapper(X[p, p])
        end
    end
end

"""
    permute_systems(X::AbstractMatrix, perm::Vector, dims::Matrix)

Permutes the order of the subsystems of the matrix `X`, which is composed by subsystems of dimensions `dims`, according to the permutation `perm`.
`dims` should be a n × 2 matrix where `dims[i, 1]` is the number of rows of subsystem i, and `dims[i, 2]` is its number of columns.
"""
function permute_systems(X::AbstractMatrix, perm::AbstractVector{<:Integer}, dims::Matrix{<:Integer})
    perm == 1:length(perm) && return X
    rowp = _idxperm(perm, dims[:, 1])
    colp = _idxperm(perm, dims[:, 2])
    return X[rowp, colp]
end
export permute_systems

"""
    permutation_matrix(dims::Union{Integer,AbstractVector}, perm::AbstractVector)

Unitary that permutes subsystems of dimension `dims` according to the permutation `perm`.
If `dims` is an Integer, assumes there are `length(perm)` subsystems of equal dimensions `dims`.
"""
function permutation_matrix(
    ::Type{T},
    dims::Union{Integer,AbstractVector{<:Integer}},
    perm::AbstractVector{<:Integer}
) where {T}
    dims = dims isa Integer ? fill(dims, length(perm)) : dims
    d = prod(dims)
    id = SA.sparse(one(T) * I, (d, d))
    return permute_systems(id, perm, dims; rows_only = true)
end
permutation_matrix(dims, perm) = permutation_matrix(Bool, dims, perm)
export permutation_matrix

@doc """
    trace_replace(X::AbstractMatrix, remove::AbstractVector, dims::AbstractVector = _equal_sizes(X))

Takes the partial trace of matrix `X` with subsystem dimensions `dims` and replace the removed subsystems by identity.
If the argument `dims` is omitted two equally-sized subsystems are assumed.
""" trace_replace(X::AbstractMatrix, remove::AbstractVector, dims::AbstractVector = _equal_sizes(X))

for (T, wrapper) ∈ [(:AbstractMatrix, :identity), (:(Hermitian), :(Hermitian)), (:(Symmetric), :(Symmetric))]
    @eval begin
        function trace_replace(
            X::$T,
            remove::AbstractVector{<:Integer},
            dims::AbstractVector{<:Integer} = _equal_sizes(X)
        )
            isempty(remove) && return X
            length(remove) == length(dims) && return SA.sparse(tr(X) * I, size(X))

            nsys = length(dims)
            nsys_rp = length(remove)
            nsys_kept = nsys - nsys_rp

            keep = _inv_ssys(remove, nsys)
            ssys_step = _step_sizes_ssys(dims)

            # TODO test if faster using views
            dims_keep = dims[keep] # The tensor dimensions of Y
            dims_rp = dims[remove] # The tensor dimensions of the traced out systems
            ssys_step_keep = ssys_step[keep]
            ssys_step_rp = ssys_step[remove]

            step_iterator_keep = _step_iterator(dims_keep, ssys_step_keep)
            step_iterator_rp = _step_iterator(dims_rp, ssys_step_rp)
            step_iterator_rp .-= 1

            #Take the partial trace
            dpt = prod(dims_keep)
            pt = zeros(eltype(X), (dpt, dpt))
            for k ∈ step_iterator_rp
                view_k_idx = k .+ step_iterator_keep
                X_ssys = @view X[view_k_idx, view_k_idx]
                pt += X_ssys
            end

            #Add the partial trace
            Y = zeros(eltype(X), size(X))  # Final output Y
            for k ∈ step_iterator_rp
                view_k_idx = k .+ step_iterator_keep
                Y[view_k_idx, view_k_idx] += pt
            end
            return $wrapper(Y)
        end
    end
end
export trace_replace

"""
    trace_replace(X::AbstractMatrix, remove::Integer, dims::AbstractVector = _equal_sizes(X))

Takes the partial trace of matrix `X` with subsystem dimensions `dims` and replace the removed subsystems by identity.
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
trace_replace(X::AbstractMatrix, remove::Integer, dims::AbstractVector{<:Integer} = _equal_sizes(X)) =
    trace_replace(X, [remove], dims)

"""
    apply_to_subsystem(
    op::AbstractMatrix,
    ρ::AbstractMatrix,
    ssys::AbstractVector,
    dims::AbstractVector = _equal_sizes(X)
Apply the operator op on the subsytems of ρ identified by ssys
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
function apply_to_subsystem(
    op::AbstractMatrix,
    ρ::AbstractMatrix,
    ssys::AbstractVector{<:Integer},
    dims::AbstractVector{<:Integer} = _equal_sizes(ρ)
)
    @assert !isempty(ssys)
    @assert prod(dims) == size(ρ)[1] "dimensions do not match with ρ"
    @assert prod(dims[ssys]) == size(op)[1] "dimensions and ssys do not match with matrix op"

    nsys = length(dims)
    keep = _inv_ssys(ssys, nsys)
    subs_step = _step_sizes_ssys(dims)

    dims_keep = dims[keep]
    dims_op = dims[ssys]
    subs_step_keep = subs_step[keep]
    subs_step_op = subs_step[ssys]

    step_iterator_ρ_keep = _step_iterator(dims_keep, subs_step_keep)
    step_iterator_ρ_keep .-= 1
    step_iterator_ρ_op = _step_iterator(dims_op, subs_step_op)
    Y = Array{eltype(ρ)}(undef, size(ρ))

    if isempty(keep)
        ρ_curr_ssys = @view ρ[step_iterator_ρ_op, step_iterator_ρ_op]
        Y[step_iterator_ρ_op, step_iterator_ρ_op] = op * ρ_curr_ssys
        return Y
    end

    for i_keep ∈ step_iterator_ρ_keep
        view_i_idx = i_keep .+ step_iterator_ρ_op
        for j_keep ∈ step_iterator_ρ_keep
            view_j_idx = j_keep .+ step_iterator_ρ_op
            ρ_curr_ssys = @view ρ[view_i_idx, view_j_idx]
            Y[view_i_idx, view_j_idx] = op * ρ_curr_ssys
        end
    end
    return Y
end

export apply_to_subsystem

"""
    apply_to_subsystem(
    op::AbstractMatrix,
    ρ::AbstractMatrix,
    ssys::Integer,
    dims::AbstractVector = _equal_sizes(ρ)
Apply the operator op on the subsytems of ρ identified by ssys
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
apply_to_subsystem(
    op::AbstractMatrix,
    ρ::AbstractMatrix,
    ssys::Integer,
    dims::AbstractVector{<:Integer} = _equal_sizes(ρ)
) = apply_to_subsystem(op, ρ, [ssys], dims)