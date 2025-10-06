@testset "Entropy               " begin
    @testset "Straight" begin
        @test binary_entropy(0) == binary_entropy(0, 2) == 0
        @test binary_entropy(1) == binary_entropy(1, 3) == 0
        @test entropy([0.0, 1.0]) == entropy([0.0, 1.0], 0.9) == 0
        @test entropy([1.0, 0.0]) == entropy([1.0, 0.0], 1.1) == 0
        for R ∈ (Float64, Double64)
            α = R(0.8)
            ρ = random_state(Complex{R}, 3, 2)
            @test isa(entropy(ρ), R)
            @test isa(entropy(ρ, α), R)
            @test entropy(ρ) ≈ entropy(ρ; base = ℯ) / log(R(2))
            @test entropy(ρ, α) ≈ entropy(ρ, α; base = ℯ) / log(R(2))
            p = random_probability(R, 3)
            @test isa(entropy(p), R)
            @test isa(entropy(p, α), R)
            @test entropy(p) ≈ entropy(p; base = ℯ) / log(R(2))
            @test entropy(p, α) ≈ entropy(p, α; base = ℯ) / log(R(2))
            @test entropy(p) ≈ entropy(Diagonal(p))
            @test entropy(p, α) ≈ entropy(Diagonal(p), α)
            q = rand(R)
            @test entropy([q, 1 - q]) ≈ binary_entropy(q)
            @test entropy([q, 1 - q], α) ≈ binary_entropy(q, α)
            @test binary_entropy(q) ≈ binary_entropy(q; base = ℯ) / log(R(2))
            @test binary_entropy(q, α) ≈ binary_entropy(q, α; base = ℯ) / log(R(2))
            @test binary_entropy(R(0.5)) == R(1)
            @test binary_entropy(R(0.75), α) ≈ (log2(3^α + 1) - 2α) / (1 - α)
        end
    end

    @testset "Relative" begin
        for R ∈ (Float64, Double64)
            α = R(0.8)
            ρ = random_state(Complex{R}, 3, 2)
            σ = random_state(Complex{R}, 3)
            @test relative_entropy(ρ, σ) ≈ relative_entropy(ρ, σ; base = ℯ) / log(R(2))
            @test relative_entropy(ρ, σ, α) ≈ relative_entropy(ρ, σ, α; base = ℯ) / log(R(2))
            id = Hermitian(R.(I(3)))
            @test relative_entropy(ρ, id) ≈ -entropy(ρ)
            @test relative_entropy(ρ, id, α) ≈ -entropy(ρ, α)
            U = random_unitary(Complex{R}, 4)
            ρ2 = Hermitian(U * [ρ zeros(3, 1); zeros(1, 3) 0] * U')
            σ2 = Hermitian(U * [σ zeros(3, 1); zeros(1, 3) 0] * U')
            @test relative_entropy(ρ, σ) ≈ relative_entropy(ρ2, σ2)
            @test relative_entropy(ρ, σ, α) ≈ relative_entropy(ρ2, σ2, α)
            p = rand(R)
            q = rand(R)
            @test binary_relative_entropy(p, q) ≈ binary_relative_entropy(p, q; base = ℯ) / log(R(2))
            @test binary_relative_entropy(p, q, α) ≈ binary_relative_entropy(p, q, α; base = ℯ) / log(R(2))
            @test binary_relative_entropy(R(1), q) ≈ -log2(q)
            @test binary_relative_entropy(R(1), q, α) ≈ -log2(q)
        end
    end

    @testset "Conditional" begin
        @test conditional_entropy(Diagonal(ones(2) / 2)) == 0.0
        @test conditional_entropy(Diagonal(ones(2) / 2), 0.9) == 0.0
        @test conditional_entropy(ones(2, 2) / 4) == 1.0
        @test conditional_entropy(ones(2, 2) / 4, 0.9) ≈ 1.0
        for R ∈ (Float64, Double64)
            α = R(0.8)
            p = random_probability(R, 2)
            pAB = [zeros(2) p]
            @test conditional_entropy(pAB) ≈ entropy(p)
            @test conditional_entropy(pAB, α) ≈ entropy(p, α)
            pAB = ones(R, 3, 2) / 6
            @test isa(conditional_entropy(pAB), R)
            @test conditional_entropy(pAB) ≈ log2(R(3))
            @test conditional_entropy(pAB, α) ≈ log2(R(3))
            pAB = reshape(random_probability(R, 6), 2, 3)
            @test conditional_entropy(pAB) ≈ conditional_entropy(pAB; base = ℯ) / log(R(2))
            @test conditional_entropy(pAB, α) ≈ conditional_entropy(pAB, α; base = ℯ) / log(R(2))
            ρAB = Diagonal(vec(pAB'))
            @test conditional_entropy(pAB) ≈ conditional_entropy(ρAB, 2, [2, 3])
            @test conditional_entropy(pAB, α) > conditional_entropy(ρAB, 2, [2, 3], α)
            @test conditional_entropy(state_phiplus(Complex{R}, 2), 2, [2, 2]) == -1
            @test conditional_entropy(state_phiplus(Complex{R}, 2), 2, [2, 2], α) ≈ -1
            ρ = random_state(Complex{R}, 6)
            @test conditional_entropy(ρ, 2, [2, 3]) ≈ conditional_entropy(ρ, 2, [2, 3]; base = ℯ) / log(R(2))
            @test conditional_entropy(ρ, 2, [2, 3], α) ≈ conditional_entropy(ρ, 2, [2, 3], α; base = ℯ) / log(R(2))
            @test conditional_entropy(ρ, [1, 2], [2, 3]) == 0
            @test conditional_entropy(ρ, [1, 2], [2, 3], α) == 0
            @test conditional_entropy(ρ, Int[], [2, 3]) ≈ entropy(ρ)
            @test conditional_entropy(ρ, Int[], [2, 3], α) ≈ entropy(ρ, α)
        end
    end
end
