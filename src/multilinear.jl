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
    _subsystems_complement(ssys::AbstractVector, nsys::Integer)

Return the complement of the set of subsystems given ; {x ∈ [1,nsys] : x ∉ ssys}
"""
function _subsystems_complement(ssys::AbstractVector{<:Integer}, nsys::Integer)
    return deleteat!(collect(1:nsys), sort(ssys))
end

"""
    _step_sizes_subsystems(dims::AbstractVector)

Return the array step_sizes s.t. 
step_sizes[j] is the step in standard index to go from tensor index 
[i₁, i₂, ..., iⱼ, ...] to tensor index [i₁, i₂, ..., iⱼ + 1, ...]
"""
function _step_sizes_subsystems(dims::AbstractVector{<:Integer})
    isempty(dims) && return Int[]
    step_sizes = Vector{Int}(undef, length(dims))
    _step_sizes_subsystems!(step_sizes, dims)
    return step_sizes
end

function _step_sizes_subsystems!(step_sizes::Vector{Int}, dims::AbstractVector{<:Integer})
    dims = Int.(dims)
    step_sizes[end] = 1
    for i ∈ length(dims)-1:-1:1
        step_sizes[i] = step_sizes[i+1] * dims[i+1]
    end
    return step_sizes
end

"""
    _step_iterator(dims::AbstractVector, step_sizes::Vector)

length(Dims) nested loops of range dims[i] each.
Returns array step_iterator s.t. 
The value at tensor index [a₁, a₂, ...] is 1 + ∑ (aᵢ - 1) * step_sizes[i]
"""
function _step_iterator(dims::AbstractVector{<:Integer}, step_sizes::Vector{Int})
    isempty(dims) && return Int[]
    step_iterator = Vector{Int}(undef, prod(dims))
    _step_iterator!(step_iterator, dims, step_sizes)
    return step_iterator
end

function _step_iterator!(step_iterator::Vector{Int}, dims::AbstractVector{<:Integer}, step_sizes::Vector{Int})
    dims = Int.(dims)
    step_sizes_idx = _step_sizes_subsystems(dims)
    _step_iterator_rec!(step_iterator, dims, step_sizes_idx, step_sizes, 1, 1, 1)
    return step_iterator
end

# Helper for _step_iterator
function _step_iterator_rec!(
    res::Vector{Int},
    dims::Vector{Int},
    step_sizes_idx::Vector{Int},
    step_sizes_res::Vector{Int},
    idx::Int,
    acc::Int,
    it::Int
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
for (T, limit, wrapper) ∈
    [(:AbstractMatrix, :dY, :identity), (:(Hermitian), :j, :(Hermitian)), (:(Symmetric), :j, :(Symmetric))]
    @eval begin
        function partial_trace(
            X::$T,
            remove::AbstractVector{<:Integer},
            dims::AbstractVector{<:Integer} = _equal_sizes(X)
        )
            isempty(remove) && return X
            length(remove) == length(dims) && return $wrapper([eltype(X)(tr(X));;])

            nsys = length(dims)

            keep = _subsystems_complement(remove, nsys)
            ssys_step = _step_sizes_subsystems(dims)

            dims_keep = dims[keep] # The tensor dimensions of Y
            dims_rm = dims[remove] # The tensor dimensions of the traced out systems

            dY = prod(dims_keep)    # Dimension of Y
            Y = Matrix{typeof(1 * X[1])}(undef, dY, dY) #hack for JuMP variables
            for i ∈ eachindex(Y)
                Y[i] = 0
            end

            ssys_step_keep = ssys_step[keep]
            ssys_step_rm = ssys_step[remove]

            step_iterator_keep = _step_iterator(dims_keep, ssys_step_keep)
            step_iterator_rm = _step_iterator(dims_rm, ssys_step_rm)
            step_iterator_rm .-= 1

            for k ∈ step_iterator_rm
                view_k_idx = k .+ step_iterator_keep
                for j ∈ 1:dY, i ∈ 1:$limit
                    Y[i, j] += X[view_k_idx[i], view_k_idx[j]]
                end
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

            nsys = length(dims)
            keep = _subsystems_complement(transp, nsys)

            dims_keep = dims[keep]
            dims_transp = dims[transp]

            keep_size = prod(dims_keep)
            transp_size = prod(dims_transp)
            prod(dims_keep) > prod(dims_transp) && return partial_transpose(transpose(X), keep, dims)

            X_size = size(X, 1)                            # Dimension of the final output Y
            Y = similar(X, (X_size, X_size))                    # Final output Y

            perm = vcat(keep, transp)
            dims_perm = vcat(dims_keep, dims_transp)

            p = sortperm(perm)
            inv_perm = collect(1:nsys)[p]
            X_perm = permute_systems(X, perm, dims)

            for j ∈ 1:transp_size:X_size-1, i ∈ 1:transp_size:X_size-1
                @views Y[i:i+transp_size-1, j:j+transp_size-1] = transpose(X_perm[i:i+transp_size-1, j:j+transp_size-1])
            end
            return $wrapper(permute_systems(Y, inv_perm, dims_perm))
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
    dims = Int.(dims)
    subsystem_og_step = _step_sizes_subsystems(dims)
    subsystem_perm_step = Vector{Int}(undef, length(dims))

    dims_view = @view dims[perm]
    step_sizes_perm = _step_sizes_subsystems(dims_view)
    @views subsystem_perm_step[perm] = step_sizes_perm[:]
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

for (T, limit, wrapper) ∈
    [(:AbstractMatrix, :dpt, :identity), (:(Hermitian), :j, :(Hermitian)), (:(Symmetric), :j, :(Symmetric))]
    @eval begin
        function trace_replace(
            X::$T,
            replace::AbstractVector{<:Integer},
            dims::AbstractVector{<:Integer} = _equal_sizes(X)
        )
            isempty(replace) && return X
            length(replace) == length(dims) && return SA.sparse(tr(X) / prod(dims[replace]) * I, size(X))

            nsys = length(dims)
            nsys_rp = length(replace)
            nsys_kept = nsys - nsys_rp

            keep = _subsystems_complement(replace, nsys)
            ssys_step = _step_sizes_subsystems(dims)

            dims_keep = dims[keep] # The tensor dimensions of Y
            dims_rp = dims[replace] # The tensor dimensions of the traced out systems
            ssys_step_keep = ssys_step[keep]
            ssys_step_rp = ssys_step[replace]

            step_iterator_keep = _step_iterator(dims_keep, ssys_step_keep)
            step_iterator_rp = _step_iterator(dims_rp, ssys_step_rp)
            step_iterator_rp .-= 1

            #Take the partial trace
            dpt = prod(dims_keep)
            pt = partial_trace(X, replace, dims)
            pt /= prod(dims_rp) # normalize for trace preservation

            #Add the partial trace
            Y = Matrix{typeof(1 * X[1])}(undef, size(X)) #hack for JuMP variables
            for i ∈ eachindex(Y)
                Y[i] = 0
            end
            for k ∈ step_iterator_rp
                view_k_idx = k .+ step_iterator_keep
                for j ∈ 1:dpt, i ∈ 1:$limit
                    Y[view_k_idx[i], view_k_idx[j]] += pt[i, j]
                end
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