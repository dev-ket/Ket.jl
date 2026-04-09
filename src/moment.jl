module Moment

# for efficiency the code assumes that the representation of each monomial or operator sequence is unique
# the functions all produce and preserve the canonical representations
# you can manually construct an ill-formed object like OperatorSequence{2,Observable}((Observable(1),Observable(1)),1)
# and get some fascinating bugs

import SparseArrays as SA

## Operator

abstract type Operator end

struct Observable <: Operator
    input::UInt8
end

Base.zero(::Type{Observable}) = Observable(0)
Base.zero(o::Operator) = zero(typeof(o))
Base.isless(o::Observable, p::Observable) = o.input < p.input

struct Projector <: Operator
    output::UInt8
    input::UInt8
end

Base.zero(::Type{Projector}) = Projector(0, 0)
function Base.isless(p::Projector, q::Projector)
    if p.input < q.input
        return true
    elseif q.input < p.input
        return false
    else
        if p.output < q.output
            return true
        else
            return false
        end
    end
end

abstract type AbstractOperatorSequence{M,O<:Operator} end

struct OperatorSequence{M,O} <: AbstractOperatorSequence{M,O}
    v::NTuple{M,O}
    length::Int8
end
struct AdjointOperatorSequence{M,O} <: AbstractOperatorSequence{M,O}
    v::NTuple{M,O}
    length::Int8
end

OperatorSequence{M,Observable}(i::Integer) where {M} =
    OperatorSequence{M,Observable}(ntuple(k -> k == 1 ? Observable(i) : zero(Observable), Val(M)), 1)
OperatorSequence{M,Projector}(a::Integer, x::Integer) where {M} =
    OperatorSequence{M,Projector}(ntuple(k -> k == 1 ? Projector(a, x) : zero(Projector), Val(M)), 1)

function Base.:(==)(v::AdjointOperatorSequence, w::OperatorSequence)
    length(v) != length(w) && return false
    for i ∈ 1:length(v)
        v[i] != w[i] && return false
    end
    return true
end
Base.:(==)(v::OperatorSequence, w::AdjointOperatorSequence) = w == v

Base.collect(os::OperatorSequence) = os
function Base.collect(os::AdjointOperatorSequence{M,O}) where {M,O}
    f(i) = i ≤ length(os) ? os[i] : zero(O)
    return OperatorSequence{M,O}(ntuple(f, Val(M)), length(os))
end

Base.length(os::AbstractOperatorSequence) = os.length
Base.iszero(os::AbstractOperatorSequence) = os.length == Int8(1) && iszero(os[1])
Base.isone(os::AbstractOperatorSequence) = iszero(length(os))
Base.lastindex(os::AbstractOperatorSequence) = Int(os.length)
Base.iterate(os::AbstractOperatorSequence, i = 1) =
    (@inline; (i - 1) % UInt < length(os) % UInt ? (@inbounds os[i], i + 1) : nothing)

Base.getindex(os::OperatorSequence, i::Int) = os.v[i]
Base.getindex(os::AdjointOperatorSequence, i::Int) = os.v[length(os)+1-i]
Base.adjoint(os::OperatorSequence) = AdjointOperatorSequence(os.v, length(os))
Base.adjoint(os::AdjointOperatorSequence) = OperatorSequence(os.v, length(os))

Base.zero(::Type{OperatorSequence{M,O}}) where {M,O} = OperatorSequence{M,O}(ntuple(i -> zero(O), Val(M)), 1)
Base.one(::Type{OperatorSequence{M,O}}) where {M,O} = OperatorSequence{M,O}(ntuple(i -> zero(O), Val(M)), 0)
Base.zero(os::AbstractOperatorSequence) = zero(typeof(os))
Base.one(os::AbstractOperatorSequence) = one(typeof(os))

function Base.:*(v::AbstractOperatorSequence{M,Observable}, w::AbstractOperatorSequence{M,Observable}) where {M}
    isone(v) && return collect(w)
    isone(w) && return collect(v)
    n = min(length(v), length(w))
    index = 0
    while v[end-index] == w[1+index]
        index += 1
        index == n && break
    end
    f = let index = index #workaround for julia#15276
        i -> if i ≤ length(v) - index
            v[i]
        elseif i ≤ length(v) + length(w) - 2 * index
            w[i-length(v)+2*index]
        else
            zero(Observable)
        end
    end
    res = ntuple(f, Val(M))
    return OperatorSequence{M,Observable}(res, length(v) + length(w) - 2 * index)
end

function Base.:*(v::AbstractOperatorSequence{M,Projector}, w::AbstractOperatorSequence{M,Projector}) where {M}
    (iszero(v) || iszero(w)) && return zero(OperatorSequence{M,Projector})
    isone(v) && return collect(w)
    isone(w) && return collect(v)
    if v[end].input != w[1].input
        f = i -> if i ≤ length(v)
            v[i]
        elseif i ≤ length(v) + length(w)
            w[i-length(v)]
        else
            zero(Projector)
        end
        return OperatorSequence{M,Projector}(ntuple(f, Val(M)), length(v) + length(w))
    else
        if v[end].output == w[1].output
            f = i -> if i ≤ length(v)
                v[i]
            elseif i ≤ length(v) + length(w) - 1
                w[i-length(v)+1]
            else
                zero(Projector)
            end
            res = ntuple(f, Val(M))
            return OperatorSequence{M,Projector}(res, length(v) + length(w) - 1)
        else
            return zero(OperatorSequence{M,Projector})
        end
    end
end

function Base.isless(v::AbstractOperatorSequence, w::AbstractOperatorSequence)
    n = min(length(v), length(w))
    for i ∈ 1:n
        if v[i] < w[i]
            return true
        elseif w[i] < v[i]
            return false
        end
    end
    if length(v) < length(w)
        return true
    end
    return false
end

## Monomial

struct Monomial{N,OS<:AbstractOperatorSequence}
    word::NTuple{N,OS}
end
Base.getindex(m::Monomial, i::Int) = getindex(m.word, i)

function Base.:*(
    m::Monomial{N,<:AbstractOperatorSequence{M,Observable}},
    n::Monomial{N,<:AbstractOperatorSequence{M,Observable}}
) where {N,M}
    return Monomial(ntuple(i -> m[i] * n[i], Val(N)))
end
function Base.:*(
    m::Monomial{N,<:AbstractOperatorSequence{M,Projector}},
    n::Monomial{N,<:AbstractOperatorSequence{M,Projector}}
) where {N,M}
    res = Monomial(ntuple(i -> m[i] * n[i], Val(N)))
    if iszero(res)
        return zero(res) #we need the representation of zero to be unique 
    else
        return res
    end
end

function Base.adjoint(m::Monomial{N,O}) where {N,O}
    return Monomial(ntuple(i -> m[i]', Val(N)))
end

Base.collect(m::Monomial{N,<:OperatorSequence}) where {N} = m
Base.collect(m::Monomial{N,<:AdjointOperatorSequence}) where {N} = Monomial(ntuple(i -> collect(m[i]), Val(N)))

function Base.isless(m::Monomial{N}, n::Monomial{N}) where {N}
    for i ∈ 1:N
        if m[i] < n[i]
            return true
        elseif n[i] < m[i]
            return false
        end
    end
    return false
end

Base.zero(::Type{Monomial{N,O}}) where {N,O} = Monomial(ntuple(i -> zero(O), Val(N)))
Base.one(::Type{Monomial{N,O}}) where {N,O} = Monomial(ntuple(i -> one(O), Val(N)))
Base.zero(m::Monomial) = zero(typeof(m))
Base.one(m::Monomial) = one(typeof(m))
function Base.iszero(m::Monomial{N}) where {N}
    for i ∈ 1:N
        if iszero(m[i])
            return true
        end
    end
    return false
end
function Base.isone(m::Monomial{N}) where {N}
    for i ∈ 1:N
        if !isone(m[i])
            return false
        end
    end
    return true
end

## pretty printing

raise(i) = ('⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹')[i+1] #fuck the Unicode Consortium
lower(i) = '₀' + i
superscript(x) = join(raise.(reverse(digits(x))))
subscript(x) = join(lower.(reverse(digits(x))))

_to_string(x::Observable) = subscript(x.input)
_to_string(x::Projector) = superscript(x.output) * subscript(x.input)

function Base.show(io::IO, x::Operator)
    if iszero(x)
        str = "0"
    else
        str = "O" * _to_string(x)
    end
    print(io, str)
end

function Base.show(io::IO, x::AbstractOperatorSequence)
    if iszero(x)
        str = "0"
    elseif isone(x)
        str = "Io"
    else
        str = ""
        for i ∈ 1:length(x)
            str *= 'O' * _to_string(x[i])
        end
    end
    print(io, str)
end

function Base.show(io::IO, x::Monomial{N,<:AbstractOperatorSequence{M}}) where {N,M}
    if iszero(x)
        str = "0"
    elseif isone(x)
        str = "I"
    else
        str = ""
        for i ∈ 1:N
            for j ∈ 1:min(M, length(x[i]))
                str *= ('@' + i) * _to_string(x[i][j])
            end
        end
    end
    print(io, str)
end

## NPA

function real_representative(m::Monomial)
    mdagger = m'
    if mdagger < m
        return collect(mdagger)
    else
        return collect(m)
    end
end

function parse_level(::Val{N}, level::Union{String,Integer}) where {N}
    if isa(level, Integer)
        @assert level > 0
        return Int(level), Vector{Int}[]
    else
        splitstr = split(replace(level, " " => ""), '+')
        levelint = parse(Int, splitstr[1])
        additional = Vector{Vector{Int}}(undef, length(splitstr) - 1)
        for i ∈ 2:length(splitstr)
            intparties = Int.(collect(splitstr[i])) .- 64
            maximum(intparties) ≤ N || error("Too many parties")
            additional[i-1] = [count(==(party), intparties) for party ∈ 1:N]
        end
        if !isempty(additional)
            all(sum.(additional) .> levelint) || error("String of additional moments already contained in level")
        end
        return levelint, additional
    end
end

function party_monomials(MonomialType::Type{Monomial{N,OperatorSequence{M,Observable}}}, party, outs, ins) where {N,M}
    monomials = Vector{MonomialType}(undef, ins[party])
    Io = one(OperatorSequence{M,Observable})
    for i ∈ 1:ins[party]
        os = OperatorSequence{M,Observable}(i)
        monomials[i] = Monomial(ntuple(k -> k == party ? os : Io, Val(N)))
    end
    return monomials
end

function party_monomials(MonomialType::Type{Monomial{N,OperatorSequence{M,Projector}}}, party, outs, ins) where {N,M}
    small_outs = outs .- 1
    monomials = Vector{MonomialType}(undef, ins[party] * small_outs[party])
    Io = one(OperatorSequence{M,Projector})
    for x ∈ 1:ins[party]
        for a ∈ 1:small_outs[party]
            os = OperatorSequence{M,Projector}(a, x)
            monomials[(x-1)*small_outs[party]+a] = Monomial(ntuple(k -> k == party ? os : Io, Val(N)))
        end
    end
    return monomials
end

function generate_sequences(
    MonomialType::Type{Monomial{N,OperatorSequence{M,O}}},
    outs::NTuple{N,<:Integer},
    ins::NTuple{N,<:Integer},
    level::Int,
    additional::Vector{Vector{Int}} = Vector{Int}[]
) where {N,M,O}
    ## level1
    mvec = [one(MonomialType)] #the first monomial must be identity
    for party ∈ 1:N
        push!(mvec, party_monomials(MonomialType, party, outs, ins)...)
    end
    ## higher
    levelrange = [2:length(mvec)]
    for k ∈ 1:level-1
        for i ∈ levelrange[1]
            for j ∈ levelrange[k]
                new_monomial = mvec[i] * mvec[j]
                if !iszero(new_monomial) && new_monomial ∉ mvec
                    push!(mvec, new_monomial)
                end
            end
        end
        push!(levelrange, levelrange[k][end]:length(mvec))
    end
    if isempty(additional)
        return mvec
    else
        partyrange = Vector{UnitRange{Int64}}(undef, N)
        rangestart = 2
        for party ∈ 1:N
            rangeend = rangestart + (outs[party] - 1) * ins[party] - 1
            partyrange[party] = rangestart:rangeend
            rangestart = rangeend + 1
        end
        for add ∈ additional
            new_monomials = [one(MonomialType)]
            for party ∈ 1:N
                while add[party] > 0
                    tempvec = MonomialType[]
                    for m ∈ new_monomials
                        for i ∈ partyrange[party]
                            new_monomial = m * mvec[i]
                            if !iszero(new_monomial) && new_monomial ∉ tempvec
                                push!(tempvec, new_monomial)
                            end
                        end
                    end
                    new_monomials = tempvec
                    add[party] -= 1
                end
            end
            for m ∈ new_monomials
                if m ∉ mvec
                    push!(mvec, m)
                end
            end
        end
        return mvec
    end
end

function spbool(i, j, n)
    colptr = Vector{Int}(undef, n + 1)
    for k ∈ 1:j
        colptr[k] = 1
    end
    for k ∈ j+1:n+1
        colptr[k] = 2
    end
    rowval = [i]
    nzval = [true]
    return SA.SparseMatrixCSC{Bool,Int}(n, n, colptr, rowval, nzval)
end

function moment_matrix(mvec::Vector{Monomial{N,O}}) where {N,O}
    n = length(mvec)
    monomial_dict = Dict{eltype(mvec),Int}()
    basis_list = Vector{SA.SparseMatrixCSC{Bool,Int}}(undef, div(n * (n + 1), 2))
    @inbounds for j ∈ 1:n
        for i ∈ 1:j
            new_monomial = real_representative(mvec[i]'mvec[j])
            if !iszero(new_monomial)
                l = length(monomial_dict)
                index = get!(monomial_dict, new_monomial, l + 1)::Int
                if index == l + 1
                    basis_list[index] = spbool(i, j, n)
                else
                    basis_list[index][i, j] = true
                end
            end
        end
    end
    resize!(basis_list, length(monomial_dict))
    return monomial_dict, basis_list
end

function explicit_moment_matrix(monomial_dict, basis_list)
    n = size(basis_list[1], 1)
    Γ = zeros(keytype(monomial_dict), n, n)
    for (monomial, index) ∈ monomial_dict
        Γi = basis_list[index]
        for j ∈ 1:size(Γi, 2)
            for k ∈ SA.nzrange(Γi, j)
                for i ∈ SA.rowvals(Γi)[k]
                    Γ[i, j] = monomial
                end
            end
        end
    end
    return Γ
end

function behaviour_operator(
    MonomialType::Type{Monomial{N,OperatorSequence{M,Observable}}},
    outs::NTuple{N,<:Integer},
    ins::NTuple{N,<:Integer}
) where {N,M}
    O = [[one(MonomialType); party_monomials(MonomialType, party, outs, ins)] for party ∈ 1:N]
    size_op = ins .+ 1
    behaviour_op = Array{MonomialType}(undef, size_op)
    for x ∈ CartesianIndices(size_op)
        behaviour_op[x] = prod(O[n][x[n]] for n ∈ 1:N)
    end
    return behaviour_op
end

function behaviour_operator(
    MonomialType::Type{Monomial{N,OperatorSequence{M,Projector}}},
    outs::NTuple{N,<:Integer},
    ins::NTuple{N,<:Integer}
) where {N,M}
    Π = [[one(MonomialType); party_monomials(MonomialType, party, outs, ins)] for party ∈ 1:N]
    cgindex(a, x) = (x .!= 1) .* (a .+ (x .- 2) .* (outs .- 1)) .+ 1
    behaviour_op = Array{MonomialType,N}(undef, (outs .- 1) .* ins .+ 1)
    for x ∈ CartesianIndices(ins .+ 1)
        for a ∈ CartesianIndices((x.I .!= 1) .* (outs .- 2) .+ 1)
            index = cgindex(a.I, x.I)
            behaviour_op[index...] = prod(Π[n][index[n]] for n ∈ 1:N)
        end
    end
    return behaviour_op
end

end #module
