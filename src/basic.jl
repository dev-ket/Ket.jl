"""
    ket([T=ComplexF64,] i::Integer, d::Integer)

Produces a ket of dimension `d` with nonzero element `i`.
"""
function ket(::Type{T}, i::Integer, d::Integer) where {T}
    psi = zeros(T, d)
    psi[i] = 1
    return psi
end
ket(i::Integer, d::Integer) = ket(ComplexF64, i, d)
export ket
"""
    ketbra(v::AbstractVector)

Produces a ketbra of vector `v`.
"""
function ketbra(v::AbstractVector)
    return LA.Hermitian(v * v')
end
export ketbra

"""
    proj([T=ComplexF64,] i::Integer, d::Integer)

Produces a projector onto the basis state `i` in dimension `d`.
"""
function proj(::Type{T}, i::Integer, d::Integer) where {T}
    p = LA.Hermitian(zeros(T, d, d))
    p[i, i] = 1
    return p
end
proj(i::Integer, d::Integer) = proj(ComplexF64, i, d)
export proj

"""
    shift([T=ComplexF64,] d::Integer, p::Integer = 1)

Constructs the shift operator X of dimension `d` to the power `p`.
"""
function shift(::Type{T}, d::Integer, p::Integer = 1) where {T}
    X = zeros(T, d, d)
    for i in 0:d-1
        X[mod(i + p, d)+1, i+1] = 1
    end
    return X
end
shift(d::Integer, p::Integer = 1) = shift(ComplexF64, d, p)
export shift

"""
    clock([T=ComplexF64,] d::Integer, p::Integer = 1)

Constructs the clock operator Z of dimension `d` to the power `p`.
"""
function clock(::Type{T}, d::Integer, p::Integer = 1) where {T}
    z = zeros(T, d)
    ω = exp(im * 2 * T(π) / d)
    for i in 0:d-1
        exponent = mod(i * p, d)
        if exponent == 0
            z[i+1] = 1
        elseif 4 * exponent == d
            z[i+1] = im
        elseif 2 * exponent == d
            z[i+1] = -1
        elseif 4 * exponent == 3 * d
            z[i+1] = -im
        else
            z[i+1] = ω^exponent
        end
    end
    return LA.Diagonal(z)
end
clock(d::Integer, p::Integer = 1) = clock(ComplexF64, d, p)
export clock

"Zeroes out real or imaginary parts of M that are smaller than `tol`"
function cleanup!(M::Array{T}; tol = Base.rtoldefault(real(T))) where {T<:Number}
    M2 = reinterpret(T, M)
    _cleanup!(M2; tol)
    return M
end
export cleanup!

function cleanup!(M::Array{T}; tol = Base.rtoldefault(T)) where {T<:Real}
    _cleanup!(M; tol)
    return M
end

function cleanup!(M::AbstractArray{T}; tol = Base.rtoldefault(real(T))) where {T<:Number}
    wrapper = Base.typename(typeof(M)).wrapper
    cleanup!(parent(M); tol)
    return wrapper(M)
end

function _cleanup!(M::AbstractArray; tol)
    return M[abs.(M).<tol] .= 0
end

function applykraus(K, M)
    return sum(LA.Hermitian(Ki * M * Ki') for Ki in K)
end
export applykraus
