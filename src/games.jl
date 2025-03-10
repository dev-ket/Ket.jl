"""
    chsh([T=Float64,] d::Integer = 2)

CHSH-d nonlocal game in probability notation. If `T` is an integer type the game is unnormalized.

Reference: Buhrman and Massar, [arXiv:quant-ph/0409066](https://arxiv.org/abs/quant-ph/0409066)
"""
function chsh(::Type{T}, d::Integer = 2) where {T}
    G = zeros(T, d, d, d, d)

    normalization = T <: Integer ? 1 : inv(T(d^2))

    for a ∈ 0:d-1, b ∈ 0:d-1, x ∈ 0:d-1, y ∈ 0:d-1
        if mod(a + b + x * y, d) == 0
            G[a+1, b+1, x+1, y+1] = normalization
        end
    end

    return G
end
chsh(d::Integer = 2) = chsh(Float64, d)
export chsh

"""
    braunsteincaves([T=Float64,] s::Integer = 3)

Braunstein-Caves nonlocal game in probability notation. Known in the computer science literature as odd cycle game. If `T` is an integer type the game is unnormalized.

References: Braunstein and Caves [doi:10.1016/0003-4916(90)90339-P](https://doi.org/10.1016/0003-4916(90)90339-P)
Cleve et al., [arXiv:quant-ph/0404076](https://arxiv.org/abs/quant-ph/0404076)
"""
function braunsteincaves(::Type{T}, s::Integer = 3) where {T}
    G = zeros(T, 2, 2, s, s)

    normalization = T <: Integer ? 1 : inv(T(2s))

    for y ∈ 0:s-1, x ∈ 0:s-1
        if x == y || (x - y - 1) % s == 0
            for a ∈ 0:1, b ∈ 0:1
                if (a + b + (x == 0) * (y == 0)) % 2 == 0
                    G[a+1, b+1, x+1, y+1] = normalization
                end
            end
        end
    end

    return G
end
braunsteincaves(s::Integer = 3) = braunsteincaves(Float64, s)
export braunsteincaves

"""
    cglmp([T=Float64,] d::Integer = 3)

CGLMP nonlocal game in probability notation. If `T` is an integer type the game is unnormalized.

References:
- Collins, Gisin, Linden, Massar, Popescu, [arXiv:quant-ph/0106024](https://arxiv.org/abs/quant-ph/0106024) (original game)
- Araújo, Hirsch, Quintino, [arXiv:2005.13418](https://arxiv.org/abs/2005.13418) (form presented here)
"""
function cglmp(::Type{T}, d::Integer = 3) where {T}
    G = zeros(T, d, d, 2, 2)

    normalization = T <: Integer ? 1 : inv(T(4 * (d - 1)))

    for a ∈ 0:d-1, b ∈ 0:d-1, x ∈ 0:1, y ∈ 0:1, k ∈ 0:d-2
        if mod(a - b, d) == mod((-1)^mod(x + y, 2) * k + x * y, d)
            G[a+1, b+1, x+1, y+1] = normalization * (d - 1 - k)
        end
    end

    return G
end
cglmp(d::Integer = 3) = cglmp(Float64, d)
export cglmp

"""
    inn22([T=Float64,] n::Integer = 3)

inn22 Bell functional in Collins-Gisin notation. Local bound 1.

Reference: Cezary Śliwa, [arXiv:quant-ph/0305190](https://arxiv.org/abs/quant-ph/0305190)
"""
function inn22(::Type{T}, n) where {T}
    C = zeros(T, n + 1, n + 1)
    for x ∈ 1:n
        for y ∈ 1:n
            if x + y ≤ n + 1
                C[x+1, y+1] = -1
            elseif x + y == n + 2
                C[x+1, y+1] = 1
            end
        end
    end
    C[1, 2] = 1
    C[2, 1] = 1
    return C
end
inn22(n::Integer = 3) = inn22(Int, n)
export inn22

"""
    gyni([T=Float64,] n::Integer)

Guess your neighbour's input nonlocal game in probability notation.
If `T` is an integer type the game is unnormalized.

Reference: Almeida et al., [arXiv:1003.3844](https://arxiv.org/abs/1003.3844)
"""
function gyni(::Type{T}, n::Integer = 3) where {T}
    G = zeros(T, ntuple(_ -> 2, 2 * n))

    normalization = T <: Integer ? 1 : inv(T(2^(n - 1)))

    nmax = n % 2 == 1 ? n : n - 1
    for x ∈ CartesianIndices(ntuple(_ -> 2, n))
        if sum(x.I[1:nmax] .- 1) % 2 == 0
            a = (x.I[2:n]..., x.I[1])
            G[a..., x] = normalization
        end
    end

    return G
end
gyni(n::Integer = 3) = gyni(Float64, n)
export gyni
