@testset "Entropy               " begin
    @testset "von Neumann" begin
        @test binary_entropy(0) == 0
        @test binary_entropy(1) == 0
        @test entropy([0.0, 1.0]) == 0
        @test entropy([1.0, 0.0]) == 0
        for R ∈ [Float64, BigFloat]
            ρ = random_state(Complex{R}, 3, 2)
            @test isa(entropy(ρ), R)
            @test entropy(ρ) ≈ entropy(ℯ, ρ) / log(R(2))
            p = random_probability(R, 3)
            @test isa(entropy(p), R)
            @test entropy(p) ≈ entropy(ℯ, p) / log(R(2))
            @test entropy(p) ≈ entropy(Diagonal(p))
            q = rand(R)
            @test entropy([q, 1 - q]) ≈ binary_entropy(q)
            @test binary_entropy(q) ≈ binary_entropy(ℯ, q) / log(R(2))
            @test binary_entropy(R(0.5)) == R(1)
        end
    end

    @testset "Relative" begin
        for R ∈ [Float64, BigFloat]
            ρ = random_state(Complex{R}, 3, 2)
            σ = random_state(Complex{R}, 3)
            @test relative_entropy(ρ, σ) ≈ relative_entropy(ℯ, ρ, σ) / log(R(2))
            id = Hermitian(Complex{R}.(I(3)))
            @test relative_entropy(ρ, id) ≈ -entropy(ρ)
            U = random_unitary(Complex{R}, 4)
            ρ2 = Hermitian(U * [ρ zeros(3, 1); zeros(1, 3) 0] * U')
            σ2 = Hermitian(U * [σ zeros(3, 1); zeros(1, 3) 0] * U')
            @test relative_entropy(ρ, σ) ≈ relative_entropy(ρ2, σ2) atol = 1.0e-8 rtol = sqrt(Base.rtoldefault(R))
            p = rand(R)
            q = rand(R)
            @test binary_relative_entropy(p, q) ≈ binary_relative_entropy(ℯ, p, q) / log(R(2))
            @test binary_relative_entropy(R(1), q) ≈ -log(2, q)
        end
    end

    @testset "Conditional" begin
        @test conditional_entropy(Diagonal(ones(2) / 2)) == 0.0
        @test conditional_entropy(ones(2, 2) / 4) == 1.0
        for R ∈ [Float64, BigFloat]
            pAB = ones(R, 3, 2) / 6
            @test isa(conditional_entropy(pAB), R)
            @test conditional_entropy(pAB) ≈ log2(R(6)) - 1
            pAB = reshape(random_probability(R, 6), 2, 3)
            @test conditional_entropy(pAB) ≈ conditional_entropy(ℯ, pAB) / log(R(2))
            rhoAB = Diagonal(vec(pAB'))
            @test conditional_entropy(pAB) ≈ conditional_entropy(rhoAB, 2, [2, 3])
            @test conditional_entropy(state_phiplus(), 2, [2, 2]) == -1
            ρ = random_state(Complex{R}, 6)
            @test conditional_entropy(ρ, 2, [2, 3]) ≈ conditional_entropy(ℯ, ρ, 2, [2, 3]) / log(R(2))
            @test conditional_entropy(ρ, [1, 2], [2, 3]) == 0
            @test conditional_entropy(ρ, Int[], [2, 3]) ≈ entropy(ρ)
        end
    end
end
