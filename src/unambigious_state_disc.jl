using JuMP
using LinearAlgebra    
import SCS
using Hypatia
using Ket

Ψ = LinearAlgebra.Hermitian([1 0; 0 0])         # |0><0|
Φ = LinearAlgebra.Hermitian(0.5 * [1 1; 1 1])   # |+><+|
ρ = [Ψ, Φ]

E= unambigious_state_discrimination(ρ)

E = [value.(e) for e in E]
println(tr(E[1] * ρ[1]))
println(tr(E[1] * ρ[2]))
println(tr(E[2] * ρ[1]))
println(tr(E[2] * ρ[2]))

