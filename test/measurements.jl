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
    @testset "Unambigious State Discrimination" begin
        for N ∈ 2:5
            ρ = [random_state(N) for i in 1:N]
            @test unambigious_state_discrimination(ρ)[N+1] ≈ I atol=1e-6
            ρ2 = [random_state(N,N-1) for i in 1:N]
            @test test_povm(unambigious_state_discrimination(ρ2))
        end
    end
end
