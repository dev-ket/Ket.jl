using Ket
using LinearAlgebra
using BenchmarkTools
using SparseArrays

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
    ssys_step = Ket._step_sizes_subsystems(dims)

    dims_keep = dims[keep] # The tensor dimensions of Y
    dims_ssys = dims[ssys] # The tensor dimensions of the traced out systems

    step_iterator_ρ_keep = Ket._step_iterator(dims_keep, ssys_step[keep])
    step_iterator_ρ_op = Ket._step_iterator(dims_ssys, ssys_step[ssys])
    step_iterator_ρ_op .-= 1

    #Add the partial trace
    Y = Matrix{typeof(1 * ρ[1])}(undef, size(ρ)) #hack for JuMP variables
    for i ∈ eachindex(Y)
        Y[i] = 0
    end

    view_i_idx = similar(step_iterator_ρ_op)
    view_j_idx = similar(step_iterator_ρ_op)

    interm = similar(op)
    for i_keep ∈ step_iterator_ρ_keep
        view_i_idx .= i_keep .+ step_iterator_ρ_op
        for j_keep ∈ step_iterator_ρ_keep
            view_j_idx .= j_keep .+ step_iterator_ρ_op
            @views mul!(interm, op, ρ[view_i_idx, view_j_idx])
            @views mul!(Y[view_i_idx, view_j_idx], interm, op')
        end
    end
    return Y
end

export apply_to_subsystem

q = rand(3:6)
dims = rand(2:6, q)
ssys = unique(rand(1:q, rand(1:q)))
dssys = prod(dims[ssys])
@info q, dims, ssys

op = rand(ComplexF64, dssys, dssys)
ρ = rand(ComplexF64, prod(dims), prod(dims))

@benchmark apply_to_subsystem(op, ρ, ssys, dims)
@benchmark apply_to_subsystem2(op, ρ, ssys, dims)

# a = apply_to_subsystem2(op, ρ, ssys, dims)
# b = apply_to_subsystem(op, ρ, ssys, dims)
# a ≈ b

d = 3^4
T = Float64
dp1 = randn(T, d - 1)
dm1 = randn(T, d - 1)
M = SparseArrays.spdiagm(-1 => dp1, 1 => dm1)
Matrix(M)

# using Ket
using Pkg
Pkg.activate(".")
__precompile__(false)
Pkg.test()