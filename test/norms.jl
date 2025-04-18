@testset "Norms                 " begin
    @testset "Square operators" begin
        for T ∈ [Float64, BigFloat]
            X = Complex{T}.([1+2im 3+4im; 5+6im 7+8im])
            svds = [√(2 * (51 + √T(2537))), √(2 * (51 - √T(2537)))]
            @test trace_norm(X) ≈ sum(svds)
            @test schatten_norm(X, 2) ≈ √T(204)
            @test schatten_norm(X, Inf) ≈ svds[1]
            p = T(ℯ)
            @test schatten_norm(X, p) ≈ sum(svds .^ p)^(1 / p)
            @test kyfan_norm(X, 2, p) ≈ schatten_norm(X, p)
            @test kyfan_norm(X, 2) ≈ schatten_norm(X, 2)
            @test kyfan_norm(X, 1) ≈ schatten_norm(X, Inf)
            @test kyfan_norm(X, 2, 1) ≈ trace_norm(X)
        end
    end

    @testset "Rectangular operators" begin
        for T ∈ [Float64, BigFloat]
            X = Complex{T}.([1+2im 3+4im; 5+6im 7+8im; 9+10im 11+12im])
            svds = [√(325 + √T(104089)), √(325 - √T(104089))]
            @test trace_norm(X) ≈ sum(svds)
            @test schatten_norm(X, 2) ≈ √T(650)
            @test schatten_norm(X, Inf) ≈ svds[1]
            p = T(ℯ)
            @test schatten_norm(X, p) ≈ sum(svds .^ p)^(1 / p)
            @test kyfan_norm(X, 2, p) ≈ schatten_norm(X, p)
            @test kyfan_norm(X, 2) ≈ schatten_norm(X, 2)
            @test kyfan_norm(X, 1) ≈ schatten_norm(X, Inf)
            @test kyfan_norm(X, 2, 1) ≈ trace_norm(X)
        end
    end
end
