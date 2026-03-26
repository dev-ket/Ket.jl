using Ket
using Ket: _rtol, _eps

using CyclotomicNumbers
using MultiFloats
Base.exp(z::Complex{Float64x2}) = Complex{Float64x2}(exp(big(z)))
Base.log(z::Complex{Float64x2}) = Complex{Float64x2}(log(big(z)))
Base.sin(x::Float64x2) = Float64x2(sin(big(x)))
Base.cos(x::Float64x2) = Float64x2(cos(big(x)))
Base.sincos(x::Float64x2) = (sin(x), cos(x))
Base.atan(x::Float64x2, y::Float64x2) = Float64x2(atan(big(x), big(y)))
using LinearAlgebra
using SparseArrays
using Test

import JuMP
import Random
import SCS

include("basic.jl")
include("channels.jl")
include("entanglement.jl")
include("entropy.jl")
include("incompatibility.jl")
include("measurements.jl")
include("multilinear.jl")
include("nonlocal.jl")
include("norms.jl")
include("parameterizations.jl")
include("random.jl")
include("states.jl")
