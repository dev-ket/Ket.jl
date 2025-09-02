"""
    _subsystems_complement(subsystems::AbstractVector, nsys::Integer)

Return the complement of the set of subsystems given; {x ∈ [1, `nsys`] : x ∉ `subsystems`}
"""
function _subsystems_complement(subsystems::AbstractVector{<:Integer}, nsys::Integer)
    return deleteat!(collect(1:nsys), sort(subsystems))
end

"""
    _step_sizes_subsystems(dims::AbstractVector)

Return the array `step_sizes` s.t. `step_sizes[j]` is the step in standard index
to go from tensor index [i₁, i₂, ..., iⱼ, ...] to tensor index [i₁, i₂, ..., iⱼ + 1, ...]
"""
function _step_sizes_subsystems(dims::AbstractVector{<:Integer})
    isempty(dims) && return similar(dims)
    step_sizes = similar(dims)
    _step_sizes_subsystems!(step_sizes, dims)
    return step_sizes
end

function _step_sizes_subsystems!(step_sizes::AbstractVector{<:Integer}, dims::AbstractVector{<:Integer})
    step_sizes[end] = 1
    for i ∈ length(dims)-1:-1:1
        step_sizes[i] = step_sizes[i+1] * dims[i+1]
    end
    return step_sizes
end

"""
    _step_iterator(dims::AbstractVector, step_sizes::Vector)

length(`dims`) nested loops of range `dims[i]` each.
Returns array s.t. the value at tensor index [a₁, a₂, ...] is 1 + ∑(aᵢ - 1) * `step_sizes[i]`
"""
function _step_iterator(dims::AbstractVector{<:Integer}, step_sizes::AbstractVector{<:Integer})
    isempty(dims) && return similar(dims)
    step_iterator = Vector{eltype(dims)}(undef, prod(dims))
    _step_iterator!(step_iterator, dims, step_sizes)
    return step_iterator
end

function _step_iterator!(
    step_iterator::AbstractVector{<:Integer},
    dims::AbstractVector{<:Integer},
    step_sizes::AbstractVector{<:Integer}
)
    step_sizes_idx = _step_sizes_subsystems(dims)
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
            subsystems_step = _step_sizes_subsystems(dims)

            dims_keep = dims[keep] # The tensor dimensions of Y
            dims_remove = dims[remove] # The tensor dimensions of the traced out systems

            dY = prod(dims_keep)    # Dimension of Y
            Y = Matrix{typeof(1 * X[1])}(undef, dY, dY) #hack for JuMP variables
            for i ∈ eachindex(Y)
                Y[i] = 0
            end

            step_iterator_keep = _step_iterator(dims_keep, subsystems_step[keep])
            step_iterator_remove = _step_iterator(dims_remove, subsystems_step[remove])
            step_iterator_remove .-= 1

            view_k_idx = similar(step_iterator_keep)
            @inbounds for k ∈ step_iterator_remove
                view_k_idx .= k .+ step_iterator_keep
                for j ∈ 1:dY, i ∈ 1:$limit
                    Y[i, j] += X[view_k_idx[i], view_k_idx[j]]
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

Takes the partial transpose of matrix `X` with subsystem dimensions `dims` over the subsystems in `transp`.
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
            keep_size > transp_size && return $wrapper(partial_transpose(transpose(X), keep, dims))

            X_size = size(X, 1)
            Y = similar(X, X_size, X_size)                 # hack to unwrap multiple layers

            perm = vcat(keep, transp)
            dims_perm = vcat(dims_keep, dims_transp)

            inv_perm = sortperm(perm)
            X_perm = permute_systems(X, perm, dims)

            @inbounds for j ∈ 1:transp_size:X_size-1, i ∈ 1:transp_size:X_size-1
                @views Y[i:i+transp_size-1, j:j+transp_size-1] .=
                    transpose(X_perm[i:i+transp_size-1, j:j+transp_size-1])
            end
            return $wrapper(permute_systems(Y, inv_perm, dims_perm))
        end
    end
end
export partial_transpose

"""
    partial_transpose(X::AbstractMatrix, transp::Integer, dims::AbstractVector = _equal_sizes(X))

Takes the partial transpose of matrix `X` with subsystem dimensions `dims` over the subsystem `transp`.
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
partial_transpose(X::AbstractMatrix, transp::Integer, dims::AbstractVector{<:Integer} = _equal_sizes(X)) =
    partial_transpose(X, [transp], dims)

"""
    _idxperm(perm::Vector, dims::Vector)

Computes the index permutation associated with permuting the subsystems of a vector with subsystem dimensions `dims` according to `perm`.
"""
function _idxperm(perm::Vector{<:Integer}, dims::Vector{<:Integer})
    p = Vector{eltype(dims)}(undef, prod(dims))
    _idxperm!(p, perm, dims)
    return p
end

function _idxperm!(p::Vector{<:Integer}, perm::Vector{<:Integer}, dims::Vector{<:Integer})
    subsystem_og_step = _step_sizes_subsystems(dims)
    subsystem_perm_step = similar(dims)

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

Takes the partial trace of matrix `X` with subsystem dimensions `dims` over the subsystems in `remove`
and replace them with normalized identity.
If the argument `dims` is omitted two equally-sized subsystems are assumed.
""" trace_replace(X::AbstractMatrix, remove::AbstractVector, dims::AbstractVector = _equal_sizes(X))

for (T, limit, wrapper) ∈
    [(:AbstractMatrix, :dim_ptX, :identity), (:(Hermitian), :j, :(Hermitian)), (:(Symmetric), :j, :(Symmetric))]
    @eval begin
        function trace_replace(
            X::$T,
            replace::AbstractVector{<:Integer},
            dims::AbstractVector{<:Integer} = _equal_sizes(X)
        )
            isempty(replace) && return X
            length(replace) == length(dims) && return $wrapper(Matrix(I * tr(X) / size(X, 1), size(X)))

            nsys = length(dims)
            keep = _subsystems_complement(replace, nsys)
            subsystems_step = _step_sizes_subsystems(dims)

            dims_keep = dims[keep] # The tensor dimensions of Y
            dims_replace = dims[replace] # The tensor dimensions of the traced out systems

            step_iterator_keep = _step_iterator(dims_keep, subsystems_step[keep])
            step_iterator_replace = _step_iterator(dims_replace, subsystems_step[replace])
            step_iterator_replace .-= 1

            #Take the partial trace
            dim_ptX = prod(dims_keep)
            ptX = parent(partial_trace(X, replace, dims)) #take the parent for efficiency

            #Add the partial trace
            Y = Matrix{typeof(1.0 * X[1])}(undef, size(X)) #hack for JuMP variables
            for i ∈ eachindex(Y)
                Y[i] = 0
            end
            view_k_idx = similar(step_iterator_keep)
            @inbounds for k ∈ step_iterator_replace
                view_k_idx .= k .+ step_iterator_keep
                for j ∈ 1:dim_ptX, i ∈ 1:$limit
                    Y[view_k_idx[i], view_k_idx[j]] += ptX[i, j]
                end
            end
            Y ./= prod(dims_replace) # normalize for trace preservation
            return $wrapper(Y)
        end
    end
end
export trace_replace
"""
    trace_replace(X::AbstractMatrix, remove::Integer, dims::AbstractVector = _equal_sizes(X))

Takes the partial trace of matrix `X` with subsystem dimensions `dims` over the subsystem `remove`
and replace it with normalized identity.
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
trace_replace(X::AbstractMatrix, remove::Integer, dims::AbstractVector{<:Integer} = _equal_sizes(X)) =
    trace_replace(X, [remove], dims)

"""
    applymap_subsystem(op::AbstractMatrix, ψ::AbstractVector, subsystems::AbstractVector, dims::AbstractVector = _equal_sizes(ρ))

Applies the operator `op` to the subsytem of `ρ` identified by `subsystems`, resulting in (op ⊗ I) * ψ.
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
function applymap_subsystem(
    op::AbstractMatrix,
    ψ::AbstractVector,
    subsystems::AbstractVector{<:Integer},
    dims::AbstractVector{<:Integer} = _equal_sizes(ψ)
)
    isempty(subsystems) && throw(ArgumentError("Subsystems vector must not be empty"))
    subsystems == 1:length(dims) && return op * ψ
    prod(dims) == length(ψ) || throw(DimensionMismatch("ψ has length $(length(ψ)), expected length $(prod(dims))"))
    prod(dims[subsystems]) == size(op, 2) ||
        throw(DimensionMismatch("op has dimensions $(size(op)), expected $((size(op,1), prod(dims[subsystems])))"))
    square_op = size(op, 1) == size(op, 2)
    contiguous_subsystems = subsystems == subsystems[1]:subsystems[end]
    if !contiguous_subsystems && !square_op
        throw(ArgumentError("op needs to be square or subsystems need to be contiguous and ordered"))
    end

    nsys = length(dims)
    keep = _subsystems_complement(subsystems, nsys)
    dims_keep = dims[keep]

    ψ_length = length(ψ)
    input_size = size(op, 2)
    output_size = size(op, 1)
    keep_size = prod(dims_keep)
    Y_length = keep_size * output_size

    perm = vcat(keep, subsystems)
    ψ_perm = permute_systems(ψ, perm, dims)

    Y_type = Base.promote_op(*, eltype(op), eltype(ψ))
    Y = Vector{Y_type}(undef, Y_length)

    if eltype(ψ) <: JuMP.AbstractJuMPScalar
        for (i_in, i_out) ∈ zip(1:input_size:1+ψ_length-input_size, 1:output_size:1+Y_length-output_size)
            @views Y[i_out:i_out+output_size-1] .= op * ψ_perm[i_in:i_in+input_size-1]
        end
    else
        for (i_in, i_out) ∈ zip(1:input_size:1+ψ_length-input_size, 1:output_size:1+Y_length-output_size)
            @views mul!(Y[i_out:i_out+output_size-1], op, ψ_perm[i_in:i_in+input_size-1])
        end
    end

    inv_perm = sortperm(perm)
    output_dims =
        contiguous_subsystems ? vcat(ones(eltype(dims), length(subsystems) - 1), [output_size]) : dims[subsystems] # either contiguous subsystems or square operators
    dims_perm_output = vcat(dims_keep, output_dims) # The dims of the subsystem when applying the inverse permutation
    return permute_systems(Y, inv_perm, dims_perm_output)
end
export applymap_subsystem
"""
    applymap_subsystem(op::AbstractMatrix, ψ::AbstractVector, subsystems::Integer, dims::AbstractVector = _equal_sizes(ρ))

Applies the operator `op` to the subsytems of `ρ` identified by `subsystems`, resulting in (op ⊗ I) * ψ.
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
applymap_subsystem(
    op::AbstractMatrix,
    ψ::AbstractVector,
    subsystems::Integer,
    dims::AbstractVector{<:Integer} = _equal_sizes(ψ)
) = applymap_subsystem(op, ψ, [subsystems], dims)
"""
    applymap_subsystem(K::AbstractVector{<:AbstractMatrix}, ρ::AbstractMatrix, subsystems::Integer, dims::AbstractVector = _equal_sizes(ρ))

Applies the Kraus operators in `K` to the subsytems of `ρ` identified by `subsystems`, resulting in ∑ᵢ(K[i] ⊗ I) * ρ * (K[i]' ⊗ I).
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
function applymap_subsystem(
    K::AbstractVector{<:AbstractMatrix},
    ρ::AbstractMatrix,
    subsystems::AbstractVector{<:Integer},
    dims::AbstractVector{<:Integer} = _equal_sizes(ρ)
)
    isempty(subsystems) && throw(ArgumentError("Subsystems vector must not be empty"))
    subsystems == 1:length(dims) && return applymap(K, ρ)
    square_kraus_ops = all([size(Ki, 1) == size(Ki, 2) for Ki ∈ K])
    contiguous_subsystems = subsystems == subsystems[1]:subsystems[end]
    if (!contiguous_subsystems && !square_kraus_ops)
        throw(ArgumentError("Kraus operators need to be square or subsystems need to be contiguous and ordered"))
    end
    nsys = length(dims)
    keep = _subsystems_complement(subsystems, nsys)

    dims_keep = dims[keep]
    input_dims = dims[subsystems]

    input_size = prod(input_dims)
    output_size = size(K[1], 1)
    keep_size = prod(dims_keep)
    ρ_size = prod(dims)
    Y_size = keep_size * output_size

    if (ρ_size, ρ_size) != size(ρ)
        throw(DimensionMismatch("ρ has dimensions $(size(ρ)), expected dimensions $((ρ_size, ρ_size))"))
    end
    if !all([size(Ki, 2) == input_size for Ki ∈ K]) || !all([size(Ki, 1) == output_size for Ki ∈ K])
        throw(DimensionMismatch("Kraus operators have invalid dimensions"))
    end

    perm = vcat(keep, subsystems)
    ρ_perm = permute_systems(ρ, perm, dims)

    kraus_type = eltype(eltype(K))
    Y_type = Base.promote_op(*, kraus_type, eltype(ρ))
    Y = Matrix{Y_type}(undef, Y_size, Y_size)
    for i ∈ eachindex(Y)
        Y[i] = 0
    end

    if eltype(ρ) <: JuMP.AbstractJuMPScalar
        for (j_in, j_out) ∈ zip(1:input_size:1+ρ_size-input_size, 1:output_size:Y_size-output_size+1),
            (i_in, i_out) ∈ zip(1:input_size:1+ρ_size-input_size, 1:output_size:Y_size-output_size+1)

            for Ki ∈ K
                @views Y[i_out:i_out+output_size-1, j_out:j_out+output_size-1] .+=
                    Ki * ρ_perm[i_in:i_in+input_size-1, j_in:j_in+input_size-1] * Ki'
            end
        end
    else
        interm = Matrix{Y_type}(undef, size(K[1]))
        for (j_in, j_out) ∈ zip(1:input_size:1+ρ_size-input_size, 1:output_size:Y_size-output_size+1),
            (i_in, i_out) ∈ zip(1:input_size:1+ρ_size-input_size, 1:output_size:Y_size-output_size+1)

            for Ki ∈ K
                @views mul!(interm, Ki, ρ_perm[i_in:i_in+input_size-1, j_in:j_in+input_size-1])
                @views mul!(Y[i_out:i_out+output_size-1, j_out:j_out+output_size-1], interm, Ki', true, true)
            end
        end
    end

    inv_perm = sortperm(perm)
    output_dims =
        contiguous_subsystems ? vcat(ones(eltype(dims), length(subsystems) - 1), [output_size]) : dims[subsystems] # either contiguous subsystems or square operators
    dims_perm_output = vcat(dims_keep, output_dims) # The dims of the subsystem when applying the inverse permutation
    result = permute_systems(Y, inv_perm, dims_perm_output)
    return _wrapper_applymap(ρ, kraus_type)(result)
end
"""
    applymap_subsystem(K::AbstractVector{<:AbstractSparseArray}, ρ::AbstractSparseArray, subsystems::AbstractVector{<:Integer}, dims::AbstractVector = _equal_sizes(ρ))

Applies the sparse Kraus operators in `K` to the subsytems of a sparse matrix `ρ` identified by `subsystems`, resulting in ∑ᵢ(K[i] ⊗ I) * ρ * (K[i]' ⊗ I).
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
function applymap_subsystem(
    K::AbstractVector{<:SA.AbstractSparseArray},
    ρ::Union{
        SA.AbstractSparseArray,
        Hermitian{<:Any,<:SA.AbstractSparseArray},
        Symmetric{<:Any,<:SA.AbstractSparseArray}
    },
    subsystems::AbstractVector{<:Integer},
    dims::AbstractVector{<:Integer} = _equal_sizes(ρ)
)
    isempty(subsystems) && throw(ArgumentError("Subsystems vector must not be empty"))
    square_kraus_ops = all([size(Ki, 1) == size(Ki, 2) for Ki ∈ K])
    contiguous_subsystems = subsystems == subsystems[1]:subsystems[end]
    if (!contiguous_subsystems && !square_kraus_ops)
        throw(ArgumentError("Kraus operators need to be square or subsystems need be contiguous and ordered"))
    end
    nsys = length(dims)
    keep = _subsystems_complement(subsystems, nsys)

    dims_keep = dims[keep]
    input_dims = dims[subsystems]

    input_size = prod(input_dims)
    output_size = size(K[1], 1)
    keep_size = prod(dims_keep)
    ρ_size = prod(dims)
    Y_size = keep_size * output_size

    if (ρ_size, ρ_size) != size(ρ)
        throw(DimensionMismatch("ρ has dimensions $(size(ρ)), expected dimensions $((ρ_size, ρ_size))"))
    end
    if !all([size(Ki, 2) == input_size for Ki ∈ K]) || !all([size(Ki, 1) == output_size for Ki ∈ K])
        throw(DimensionMismatch("Kraus operators have invalid dimensions"))
    end

    perm = vcat(keep, subsystems)
    ρ_perm = permute_systems(ρ, perm, dims)

    kraus_type = eltype(eltype(K))
    Y_type = Base.promote_op(*, kraus_type, eltype(ρ))
    Y = SA.spzeros(Y_type, Y_size, Y_size)
    spI = SA.sparse(I, keep_size, keep_size)
    if eltype(ρ) <: JuMP.AbstractJuMPScalar
        for Ki ∈ K
            k_kron = kron(spI, Ki)
            Y .+= k_kron * ρ_perm * k_kron'
        end
    else
        temp = SA.spzeros(Y_type, output_size * keep_size, input_size * keep_size)
        for Ki ∈ K
            k_kron = kron(spI, Ki)
            mul!(temp, k_kron, ρ_perm)
            mul!(Y, temp, k_kron', true, true)
        end
    end

    inv_perm = sortperm(perm)
    output_dims =
        contiguous_subsystems ? vcat(ones(eltype(dims), length(subsystems) - 1), [output_size]) : dims[subsystems] # either contiguous subsystems or square operators
    dims_perm_output = vcat(dims_keep, output_dims) # The dims of the subsystem when applying the inverse permutation
    result = permute_systems(Y, inv_perm, dims_perm_output)
    return _wrapper_applymap(ρ, kraus_type)(result)
end
"""
    applymap_subsystem(K::AbstractVector{<:AbstractMatrix}, ρ::AbstractMatrix, subsystems::Integer, dims::AbstractVector = _equal_sizes(ρ))

Applies the Kraus operators in `K` to the subsytem of `ρ` identified by `subsystems`, resulting in ∑ᵢ(K[i] ⊗ I) * ρ * (K[i]' ⊗ I).
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
function applymap_subsystem(
    K::AbstractVector{<:AbstractMatrix},
    ρ::AbstractMatrix,
    subsystems::Integer,
    dims::AbstractVector{<:Integer} = _equal_sizes(ρ)
)
    applymap_subsystem(K, ρ, [subsystems], dims)
end
