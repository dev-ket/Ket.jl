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
    isempty(dims) && return eltype(dims)[]
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

length(Dims) nested loops of range dims[i] each.
Returns array step_iterator s.t. 
The value at tensor index [a₁, a₂, ...] is 1 + ∑ (aᵢ - 1) * step_sizes[i]
"""
function _step_iterator(dims::AbstractVector{<:Integer}, step_sizes::AbstractVector{<:Integer})
    isempty(dims) && return eltype(dims)[]
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
            ssys_step = _step_sizes_subsystems(dims)

            dims_keep = dims[keep] # The tensor dimensions of Y
            dims_remove = dims[remove] # The tensor dimensions of the traced out systems

            dY = prod(dims_keep)    # Dimension of Y
            Y = Matrix{typeof(1 * X[1])}(undef, dY, dY) #hack for JuMP variables
            for i ∈ eachindex(Y)
                Y[i] = 0
            end

            step_iterator_keep = _step_iterator(dims_keep, ssys_step[keep])
            step_iterator_remove = _step_iterator(dims_remove, ssys_step[remove])
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
            prod(dims_keep) > prod(dims_transp) && return partial_transpose(transpose(X), keep, dims)

            X_size = size(X, 1)
            Y = similar(X, X_size, X_size)                 # hack to unwrap multiple layers

            perm = vcat(keep, transp)
            dims_perm = vcat(dims_keep, dims_transp)

            p = sortperm(perm)
            inv_perm = collect(1:nsys)[p]
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
            ssys_step = _step_sizes_subsystems(dims)

            dims_keep = dims[keep] # The tensor dimensions of Y
            dims_replace = dims[replace] # The tensor dimensions of the traced out systems

            step_iterator_keep = _step_iterator(dims_keep, ssys_step[keep])
            step_iterator_replace = _step_iterator(dims_replace, ssys_step[replace])
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
apply_to_subsystem(
op::AbstractMatrix,
ψ::AbstractVector,
ssys::AbstractVector,
dims::AbstractVector = _equal_sizes(ρ)
outssys_to_end::Bool = false
Apply the operator `op` on the subsytems of `ρ` identified by `ssys`
(op ⊗ I) * ψ
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
function apply_to_subsystem(
    op::AbstractMatrix,
    ψ::AbstractVector,
    ssys::AbstractVector{<:Integer},
    dims::AbstractVector{<:Integer} = _equal_sizes(ψ)
)
    @assert !isempty(ssys)
    @assert prod(dims) == length(ψ) "dimensions do not match with ψ"
    @assert prod(dims[ssys]) == size(op, 2) "dimensions and ssys do not match with matrix op"

    square_op = size(op, 1) == size(op, 2)
    ctg_ssys = length(ssys) == ssys[end] - ssys[1] + 1 && ssys == ssys[1]:ssys[end]
    @assert ctg_ssys || square_op "Operator needs to be square or ssys need to be contiguous and ordered"

    nsys = length(dims)
    keep = _subsystems_complement(ssys, nsys)

    dims_keep = dims[keep]

    ψ_length = length(ψ)
    input_size = size(op, 2)
    output_size = size(op, 1)
    keep_size = prod(dims_keep)
    Y_length = keep_size * output_size

    perm = vcat(keep, ssys)
    ψ_perm = permute_systems(ψ, perm, dims)

    Y = Vector{typeof(1 * ψ[1])}(undef, Y_length) #hack for JuMP variables

    if eltype(ψ) <: JuMP.AbstractJuMPScalar
        for (i_in, i_out) ∈ zip(1:input_size:ψ_length-1, 1:output_size:Y_length-1)
            @views Y[i_out:i_out+output_size-1] .= op * ψ_perm[i_in:i_in+input_size-1]
        end
    else
        for (i_in, i_out) ∈ zip(1:input_size:ψ_length-1, 1:output_size:Y_length-1)
            @views mul!(Y[i_out:i_out+output_size-1], op, ψ_perm[i_in:i_in+input_size-1])
        end
    end

    inv_perm = sortperm(perm)
    output_dims = ctg_ssys ? vcat(ones(eltype(dims), length(ssys) - 1), [output_size]) : dims[ssys] # either contiguous ssys or square operators
    dims_perm_output = vcat(dims_keep, output_dims) # The dims of the subsystem when applying the inverse permutation
    return permute_systems(Y, inv_perm, dims_perm_output)
end

export apply_to_subsystem

"""
apply_to_subsystem(
op::AbstractMatrix,
ψ::AbstractVector,
ssys::Integer,
dims::AbstractVector = _equal_sizes(ρ)
Apply the operator `op` on the subsytems of `ρ` identified by `ssys`
(op ⊗ I) * ψ
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
apply_to_subsystem(
    op::AbstractMatrix,
    ψ::AbstractVector,
    ssys::Integer,
    dims::AbstractVector{<:Integer} = _equal_sizes(ψ)
) = apply_to_subsystem(op, ψ, [ssys], dims)

"""
apply_to_subsystem(
kraus::AbstractVector{<:AbstractMatrix},
ρ::AbstractMatrix,
ssys::AbstractVector,
dims::AbstractVector = _equal_sizes(ρ)
outssys_to_end::Bool = false
Apply the operators `kraus` on the subsytems of `ρ` identified by `ssys`
∑(Kᵢ ⊗ I) * ρ * (Kᵢ ⊗ I)†
For each input subsystem `i` of size `dims[ssys[i]]` they are mapped to a subsystem of size `output_dims[i]`
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
function apply_to_subsystem(
    kraus::AbstractVector{<:AbstractMatrix},
    ρ::AbstractMatrix,
    ssys::AbstractVector{<:Integer},
    dims::AbstractVector{<:Integer} = _equal_sizes(ρ),
    outssys_to_end::Bool = false
)
    @assert !isempty(ssys)
    square_kraus_ops = all([size(k, 1) == size(k, 2) for k ∈ kraus])
    ctg_ssys = length(ssys) == ssys[end] - ssys[1] + 1 && ssys == ssys[1]:ssys[end]
    @assert ctg_ssys || square_kraus_ops "Kraus operators need to be square or ssys need to be contiguous and ordered"

    nsys = length(dims)
    keep = _subsystems_complement(ssys, nsys)

    dims_keep = dims[keep]
    input_dims = dims[ssys]

    input_size = prod(input_dims) #input size of Kraus
    output_size = size(kraus[1], 1)
    keep_size = prod(dims_keep)
    ρ_size = size(ρ, 1)
    Y_size = keep_size * output_size

    @assert prod(dims) == ρ_size "dimensions do not match with ρ"
    @assert all([size(k, 2) == input_size for k ∈ kraus]) "dimensions do not match with kraus operators"
    @assert all([size(k, 1) == output_size for k ∈ kraus]) "dimensions do not match with kraus operators"

    perm = vcat(keep, ssys)
    ρ_perm = permute_systems(ρ, perm, dims)

    Y = Matrix{typeof(1 * ρ[1])}(undef, Y_size, Y_size) #hack for JuMP variables
    for i ∈ eachindex(Y) # Init with 0s
        Y[i] = 0
    end

    if eltype(ρ) <: JuMP.AbstractJuMPScalar
        for (j_in, j_out) ∈ zip(1:input_size:ρ_size-1, 1:output_size:Y_size-1),
            (i_in, i_out) ∈ zip(1:input_size:ρ_size-1, 1:output_size:Y_size-1)

            for kraus_op ∈ kraus
                @views Y[i_out:i_out+output_size-1, j_out:j_out+output_size-1] .+=
                    kraus_op * ρ_perm[i_in:i_in+input_size-1, j_in:j_in+input_size-1] * kraus_op'
            end
        end
    else
        interm = Matrix{eltype(kraus[1])}(undef, size(kraus[1]))
        for (j_in, j_out) ∈ zip(1:input_size:ρ_size-1, 1:output_size:Y_size-1),
            (i_in, i_out) ∈ zip(1:input_size:ρ_size-1, 1:output_size:Y_size-1)

            for kraus_op ∈ kraus
                @views mul!(interm, kraus_op, ρ_perm[i_in:i_in+input_size-1, j_in:j_in+input_size-1])
                @views Y[i_out:i_out+output_size-1, j_out:j_out+output_size-1] .+= interm * kraus_op'
            end
        end
    end

    inv_perm = sortperm(perm)
    output_dims = ctg_ssys ? vcat(ones(eltype(dims), length(ssys) - 1), [output_size]) : dims[ssys] # either contiguous ssys or square operators
    dims_perm_output = vcat(dims_keep, output_dims) # The dims of the subsystem when applying the inverse permutation
    return permute_systems(Y, inv_perm, dims_perm_output)
end

"""
apply_to_subsystem(
kraus::AbstractVector{<:AbstractMatrix},
ρ::AbstractMatrix,
ssys::AbstractVector,
dims::AbstractVector = _equal_sizes(ρ)
outssys_to_end::Bool = false
Apply the operators `kraus` on the subsytems of `ρ` identified by `ssys`
∑(Kᵢ ⊗ I) * ρ * (Kᵢ ⊗ I)†
If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
function apply_to_subsystem(
    kraus::AbstractVector{<:SA.AbstractSparseArray},
    ρ::SA.AbstractSparseArray,
    ssys::AbstractVector{<:Integer},
    dims::AbstractVector{<:Integer} = _equal_sizes(ρ)
)
    @assert !isempty(ssys)
    square_kraus_ops = all([size(k, 1) == size(k, 2) for k ∈ kraus])
    ctg_ssys = length(ssys) == ssys[end] - ssys[1] + 1 && ssys == ssys[1]:ssys[end]
    @assert ctg_ssys || square_kraus_ops "Kraus operators need to be square or ssys need to be contiguous and ordered"

    nsys = length(dims)
    keep = _subsystems_complement(ssys, nsys)

    dims_keep = dims[keep]
    input_dims = dims[ssys]

    input_size = prod(input_dims) #input size of Kraus
    output_size = size(kraus[1], 1)
    keep_size = prod(dims_keep)
    ρ_size = size(ρ, 1)
    Y_size = keep_size * output_size

    @assert prod(dims) == ρ_size "dimensions do not match with ρ"
    @assert all([size(k, 2) == input_size for k ∈ kraus]) "dimensions do not match with kraus operators"
    @assert all([size(k, 1) == output_size for k ∈ kraus]) "dimensions do not match with kraus operators"

    perm = vcat(keep, ssys)
    ρ_perm = permute_systems(ρ, perm, dims)

    Y = Matrix{typeof(1 * ρ[1])}(undef, Y_size, Y_size) #hack for JuMP variables
    for i ∈ eachindex(Y) # Init with 0s
        Y[i] = 0
    end

    interm = similar(ρ, output_size * keep_size, input_size * keep_size)
    for k ∈ kraus
        k_kron = kron(I(prod(dims_keep)), k)
        @views mul!(interm, k_kron, ρ_perm)
        Y .+= interm * k_kron'
    end

    Y = SA.sparse(Y)

    inv_perm = sortperm(perm)
    output_dims = ctg_ssys ? vcat(ones(eltype(dims), length(ssys) - 1), [output_size]) : dims[ssys] # either contiguous ssys or square operators
    dims_perm_output = vcat(dims_keep, output_dims) # The dims of the subsystem when applying the inverse permutation
    return permute_systems(Y, inv_perm, dims_perm_output)
end
"""
apply_to_subsystem(
kraus::AbstractVector{<:AbstractMatrix},
ρ::AbstractMatrix,
ssys::Integer,
dims::AbstractVector = _equal_sizes(ρ)
outssys_to_end::Bool = false
Apply the operators `kraus` on the subsytem of `ρ` identified by `ssys`
∑(Kᵢ ⊗ I) * ρ * (Kᵢ ⊗ I)†
The subsystem identified by `ssys` of size `dims[ssys]` is mapped to a subsystem of size `output_dims[ssys]`
"""
function apply_to_subsystem(
    kraus::AbstractVector{<:AbstractMatrix},
    ρ::AbstractMatrix,
    ssys::Integer,
    dims::AbstractVector{<:Integer} = _equal_sizes(ρ)
)
    apply_to_subsystem(kraus, ρ, [ssys], dims)
end