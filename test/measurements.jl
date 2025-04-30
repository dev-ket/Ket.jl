@testset "Measurements          " begin
    @testset "POVMs" begin
        for T ∈ [Float64, Double64, Float128, BigFloat]
            E = random_povm(Complex{T}, 2, 3)
            V = dilate_povm(E)
            @test [V' * kron(I(2), proj(i, 3)) * V for i ∈ 1:3] ≈ E
            e = sic_povm(Complex{T}, 2)
            V = dilate_povm(e)
            @test [V' * proj(i, 4) * V for i ∈ 1:4] ≈ [ketbra(e[i]) for i ∈ 1:4]
        end
    end
    @testset "SIC POVMs" begin
        for T ∈ [Float64, Double64, Float128, BigFloat], d ∈ 1:9
            @test test_sic(sic_povm(Complex{T}, d))
        end
    end
    @testset "MUBs" begin
        for T ∈ [Int8, Int64, BigInt]
            @test test_mub(mub(T(6)))
        end
        for R ∈ [Float64, Double64, Float128, BigFloat]
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
        @test state_discrimination_min_error([[1 0; 0 0],0.5 * [1 1; 1 1]])[2] ≈ (2+sqrt(2))/4 atol=1e-6 
        @test state_discrimination_min_error([[1 0; 0 0], [0 0; 0 1]])[2] ≈ 1.0 atol=1e-6
        @test state_discrimination_min_error([[1 0 0; 0 0 0; 0 0 0], [1 0 0; 0 1 0; 0 0 1],[0 0 0; 0 0 0; 0 0 1]])[2] ≈ 1.0 atol=1e-6
        for R ∈ (Float64, Double64, BigFloat), T ∈ (R, Complex{R})
            N = 3
            ρ = [random_state(T,N) for i in 1:N]
            E = state_discrimination_min_error(ρ)[1]
            @test sum(E) ≈ I atol=1e-5
            @test all(ishermitian.(E))
        end
    end

    @testset "Pretty good POVM" begin
        ρ = [1/2 * [1 1; 1 1],[1 0; 0 0]]
        E = pretty_good_povm(ρ)
        @test tr(E[1]*ρ[1]) ≈ (2+sqrt(2))/4 atol=1e-6

        for R ∈ (Float64, Double64, BigFloat), T ∈ (R, Complex{R})
            N = 3
            ρ = [random_state(T,N) for i in 1:N]
            E = pretty_good_povm(ρ)
            @test sum(E) ≈ I
        end
    end
end
