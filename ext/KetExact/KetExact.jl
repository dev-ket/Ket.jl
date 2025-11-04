module KetExact

import CyclotomicNumbers as CN
import Ket
import LinearAlgebra as LA

Base.complex(::Type{CN.Cyc{R}}) where {R<:Real} = CN.Cyc{R}

Ket._root_unity(::Type{CN.Cyc{R}}, n::Integer) where {R<:Real} = CN.E(Int(n))
Ket._sqrt(::Type{CN.Cyc{R}}, n::Integer) where {R<:Real} = CN.root(Int(n))
Ket._rtol(::Type{CN.Cyc{R}}) where {R<:Real} = sqrt(_eps(R))
Ket._eps(::Type{CN.Cyc{R}}) where {R<:Real} = R(0)
Ket._eps(::Type{CN.Cyc{R}}) where {R<:AbstractFloat} = eps(R)

LA.norm(v::AbstractVector{CN.Cyc{R}}) where {R<:Real} = Ket._sqrt(CN.Cyc{R}, Int(sum(abs2, v)))

end # module
