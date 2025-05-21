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
