module KetExact

import CyclotomicNumbers as CN
import Ket
import LinearAlgebra as LA

const TExact = CN.Cyc{Rational{BigInt}}
export TExact

Ket._root_unity(::Type{CN.Cyc{R}}, n::Integer) where {R<:Real} = CN.E(Int64(n))
Ket._sqrt(::Type{CN.Cyc{R}}, n::Integer) where {R<:Real} = CN.root(Int64(n))
Ket._rtol(::Type{CN.Cyc{R}}) where {R<:Real} = sqrt(_eps(R))
Ket._eps(::Type{CN.Cyc{R}}) where {R<:Real} = R(0)
Ket._eps(::Type{CN.Cyc{R}}) where {R<:AbstractFloat} = eps(R)

LA.norm(v::AbstractVector{CN.Cyc{R}}) where {R<:Real} = Ket._sqrt(CN.Cyc{R}, Int64(sum(abs2, v)))

end # module
