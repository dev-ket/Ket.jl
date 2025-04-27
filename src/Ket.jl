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
const JuMPReal = Union{JuMP.GenericVariableRef{<:Real},JuMP.GenericAffExpr{<:Real},JuMP.GenericQuadExpr{<:Real}}

"""
    Measurement{T}

Alias for `Vector{Hermitian{T,Matrix{T}}}`
"""
const Measurement{T} = Vector{Hermitian{T,Matrix{T}}}
export Measurement

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
include("states.jl")
include("tsirelson.jl")

end # module Ket
