"""
    schatten_norm(X::AbstractMatrix, p::Real)

Computes Schatten `p`-norm of matrix `X`.
"""
function schatten_norm(X::AbstractMatrix, p::Real)
    if p == 2
        return norm(X)
    elseif p == Inf
        return opnorm(X)
    else
        sv = svdvals(X)
        return norm(sv, p)
    end
end
export schatten_norm

"""
    trace_norm(X::AbstractMatrix)

Computes trace norm of matrix `X`.
"""
function trace_norm(X::AbstractMatrix)
    return schatten_norm(X, 1)
end
export trace_norm

"""
    kyfan_norm(X::AbstractMatrix, k::Integer, p::Real = 2)

Computes Ky-Fan (`k`,`p`) norm of matrix `X`.
"""
function kyfan_norm(X::AbstractMatrix, k::Integer, p::Real = 2)
    sv = svdvals(X)
    return norm(sv[1:k], p)
end
export kyfan_norm
