using Ket
using LinearAlgebra
using BenchmarkTools

function apply_to_subsystem2(
    op::AbstractMatrix,
    ρ::AbstractMatrix,
    ssys::AbstractVector{<:Integer},
    dims::AbstractVector{<:Integer} = Ket._equal_sizes(ρ)
)
    @assert !isempty(ssys)
    @assert prod(dims) == size(ρ)[1] "dimensions do not match with ρ"
    @assert prod(dims[ssys]) == size(op)[1] "dimensions and ssys do not match with matrix op"

    nsys = length(dims)
    keep = Ket._subsystems_complement(ssys, nsys)
    subs_step = Ket._step_sizes_subsystems(dims)

    dims_keep = dims[keep]
    dims_op = dims[ssys]
    subs_step_keep = subs_step[keep]
    subs_step_op = subs_step[ssys]

    step_iterator_ρ_keep = Ket._step_iterator(dims_keep, subs_step_keep)
    step_iterator_ρ_keep .-= 1
    step_iterator_ρ_op = Ket._step_iterator(dims_op, subs_step_op)
    Y = Array{eltype(ρ)}(undef, size(ρ))

    if isempty(keep)
        ρ_curr_ssys = @view ρ[step_iterator_ρ_op, step_iterator_ρ_op]
        Y[step_iterator_ρ_op, step_iterator_ρ_op] = op * ρ_curr_ssys
        return Y
    end
    view_i_idx = similar(step_iterator_ρ_op)
    view_j_idx = similar(step_iterator_ρ_op)
    for i_keep ∈ step_iterator_ρ_keep
        view_i_idx .= i_keep .+ step_iterator_ρ_op
        for j_keep ∈ step_iterator_ρ_keep
            view_j_idx .= j_keep .+ step_iterator_ρ_op
            ρ_curr_ssys = @view ρ[view_i_idx, view_j_idx]
            Y[view_i_idx, view_j_idx] .= op * ρ_curr_ssys * op'
        end
    end
    return Y
end

q = rand(3:6)
dims = rand(2:6, q)
ssys = unique(rand(1:q, rand(1:q)))
dssys = prod(dims[ssys])
@info q, dims, ssys

op = rand(ComplexF64, dssys, dssys)
ρ = rand(ComplexF64, prod(dims), prod(dims))

@benchmark apply_to_subsystem(op, ρ, ssys, dims)
@benchmark apply_to_subsystem2(op, ρ, ssys, dims)

a = apply_to_subsystem2(op, ρ, ssys, dims)
b = apply_to_subsystem(op, ρ, ssys, dims)
a ≈ b