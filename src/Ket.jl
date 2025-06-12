"""
Toolbox for quantum information, nonlocality, and entanglement.
"""
module Ket

using LinearAlgebra

import Base.AbstractVecOrTuple
import Combinatorics
import Dualization
import GenericLinearAlgebra
import Hypatia, Hypatia.Cones
import JuMP
import Nemo
import QuantumNPA
import SparseArrays as SA
import FixedSizeArrays

const FArray = FixedSizeArrays.FixedSizeArrayDefault
const FVector = FixedSizeArrays.FixedSizeVectorDefault
const FMatrix = FixedSizeArrays.FixedSizeMatrixDefault

for (fname, felt) in ((:fzeros, :zero), (:fones, :one))
    @eval begin
        $fname(dims::Base.DimOrInd...) = $fname(dims)
        $fname(::Type{T}, dims::Base.DimOrInd...) where {T} = $fname(T, dims)
        $fname(dims::Tuple{Vararg{Base.DimOrInd}}) = $fname(Float64, dims)
        $fname(::Type{T}, dims::NTuple{N, Union{Integer, Base.OneTo}}) where {T,N} = $fname(T, map(to_dim, dims))
        function $fname(::Type{T}, dims::NTuple{N, Integer}) where {T,N}
            a = FArray{T,N}(undef, dims)
            fill!(a, $felt(T))
            return a
        end
        function $fname(::Type{T}, dims::Tuple{}) where {T}
            a = FArray{T}(undef)
            fill!(a, $felt(T))
            return a
        end
        function $fname(::Type{T}, dims::NTuple{N, Base.DimOrInd}) where {T,N}
            a = similar(FArray{T,N}, dims)
            fill!(a, $felt(T))
            return a
        end
    end
end

const MOI = JuMP.MOI

"""
    Measurement{T}

Alias for `Vector{Hermitian{T,Matrix{T}}}`
"""
const Measurement{T} = Vector{Hermitian{T,Matrix{T}}}
export Measurement

#extract from T the kind of float to be used in conic solvers
_solver_type(::Type{T}) where {T<:Number} = float(real(T))

_rtol(::Type{T}) where {T<:Number} = Base.rtoldefault(real(T))
function _eps(::Type{T}) where {T<:Number}
    if real(T) <: AbstractFloat
        return eps(real(T))
    else
        return zero(real(T))
    end
end

include("basic.jl")
include("channels.jl")
include("entanglement.jl")
include("entropy.jl")
include("games.jl")
include("incompatibility.jl")
include("measurements.jl")
include("multilinear.jl")
include("nonlocal.jl")
include("norms.jl")
include("parameterizations.jl")
include("random.jl")
include("seesaw.jl")
include("sic-povm.jl")
include("states.jl")
include("tsirelson.jl")

end # module Ket
