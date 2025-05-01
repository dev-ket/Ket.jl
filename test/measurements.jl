@testset "Measurements          " begin
    @testset "POVMs" begin
        for T ∈ [Float64, BigFloat]
            E = random_povm(Complex{T}, 2, 3)
            V = dilate_povm(E)
            @test [V' * kron(I(2), proj(i, 3)) * V for i ∈ 1:3] ≈ E
            e = sic_povm(Complex{T}, 2)
            V = dilate_povm(e)
            @test [V' * proj(i, 4) * V for i ∈ 1:4] ≈ [ketbra(e[i]) for i ∈ 1:4]
        end
    end
    @testset "SIC POVMs" begin
        for T ∈ [Float64, BigFloat], d ∈ 1:9
            @test test_sic(sic_povm(Complex{T}, d))
        end
    end
    @testset "MUBs" begin
        for T ∈ [Int8, Int64, BigInt]
            @test test_mub(mub(T(6)))
        end
        for R ∈ [Float64, BigFloat]
            T = Complex{R}
            @test test_mub(mub(T, 2))
            @test test_mub(mub(T, 3))
            @test test_mub(mub(T, 4))
            @test test_mub(mub(T, 6))
            @test test_mub(mub(T, 9))
        end
        for T ∈ [Int64, Int128, BigInt]
            @test test_mub(broadcast.(Rational{T}, mub(Cyc{Rational{T}}, 4, 2)))
            @test test_mub(broadcast.(Complex{Rational{T}}, mub(Cyc{Rational{T}}, 4)))
        end
        @test test_mub(mub(Cyc{Rational{BigInt}}, 5, 5, 7)) # can access beyond the number of combinations
    end

    @testset "State Discrimination" begin
        N = 3
        ρ = [random_state(2) for _ in 1:N]
        p, E = discrimination_min_error(ρ)
        @test p ≈ sum(dot.(ρ, E))/N
        for R ∈ (Float64, Double64), T ∈ (R, Complex{R})
            ρ = [random_state(T,3) for _ in 1:2]
            successProb = discrimination_min_error(ρ)[1]
            @test (0.5 + 0.25 * sum(svdvals(ρ[1] - ρ[2]))) ≈ successProb atol=1e-6
        end
    end

    @testset "Pretty good measurements" begin
        ρ = [1/2 * [1 1; 1 1],[1 0; 0 0]]
        E = pretty_good_measurement(ρ)
        @test tr(E[1]*ρ[1]) ≈ (2+sqrt(2))/4 atol=1e-6

        for R ∈ (Float64, Double64), T ∈ (R, Complex{R})
            N = 3
            ρ = [random_state(T,N) for i in 1:N]
            E = pretty_good_measurement(ρ)
            @test test_povm(E)
        end
    end
end
