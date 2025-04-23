@testset "Entanglement          " begin
    @testset "Schmidt decomposition" begin
        for R ∈ (Float64, BigFloat)
            T = Complex{R}
            ψ = random_state_ket(T, 6)
            λ, U, V = schmidt_decomposition(ψ, [2, 3])
            @test vec(Diagonal(λ)) ≈ kron(U', V') * ψ
            ψ = random_state_ket(T, 4)
            λ, U, V = schmidt_decomposition(ψ)
            @test vec(Diagonal(λ)) ≈ kron(U', V') * ψ
        end
    end
    @testset "Entanglement entropy" begin
        for R ∈ (Float64, Double64), T ∈ (R, Complex{R}) #BigFloat takes too long
            Random.seed!(8) #makes all states entangled
            ψ = random_state_ket(T, 6)
            @test entanglement_entropy(ψ, [2, 3]) ≈ entanglement_entropy(white_noise!(ketbra(ψ),0.9999), [2, 3])[1] atol = 1e-3 rtol = 1e-3
            ρ = random_state(T, 4)
            h, σ = entanglement_entropy(ρ)
            @test Ket._test_entanglement_entropy_qubit(h, ρ, σ)
        end
    end
    @testset "DPS hierarchy" begin
        for R ∈ (Float64, Double64), T ∈ (R, Complex{R})
            # outer DPS:
            ρ = state_ghz(T, 2, 2)
            s, W = entanglement_robustness(ρ; noise = "white")
            @test eltype(W) == T
            @test s ≈ 0.5 atol = 1.0e-5 rtol = 1.0e-5
            @test dot(ρ, W) ≈ -s atol = 1.0e-5 rtol = 1.0e-5
            s, W = entanglement_robustness(ρ; noise = "general")
            @test eltype(W) == T
            @test s ≈ 0.25 atol = 1.0e-5 rtol = 1.0e-5
            @test dot(ρ, W) ≈ -s atol = 1.0e-5 rtol = 1.0e-5
            # inner DPS:
            s, W = entanglement_robustness(ρ, [2, 2], 2; ppt = false, inner = true)
            @test eltype(W) == T
            @test s ≈ 0.5 atol = 1.0e-5 rtol = 1.0e-5
            @test dot(ρ, W) ≈ -s atol = 1.0e-5 rtol = 1.0e-5
        end
        for R ∈ (Float32, Float64, BigFloat)
            # Legendre
            @test Ket._jacobi_polynomial_zeros(R, 1, 0, 0) ≈ [0]
            @test Ket._jacobi_polynomial_zeros(R, 2, 0, 0) ≈ [-1 / sqrt(3), 1 / sqrt(3)]
            #Chebyshev
            @test Ket._jacobi_polynomial_zeros(R, 1, -1/2, -1/2) ≈ [0]
            @test Ket._jacobi_polynomial_zeros(R, 2, -1/2, -1/2) ≈ [-1 / sqrt(2), 1 / sqrt(2)]
            @test Ket._jacobi_polynomial_zeros(R, 3, -1/2, -1/2) ≈ [-sqrt(3) / 2, 0, sqrt(3) / 2]
        end
        d = 3
        @test isapprox(schmidt_number(state_ghz(ComplexF64, d, 2), 2), 1 / 15, atol = 1.0e-3, rtol = 1.0e-3)
        @test isapprox(
            schmidt_number(state_ghz(Float64, d, 2), 2, [d, d], 2; solver = SCS.Optimizer),
            1 / 15,
            atol = 1.0e-3,
            rtol = 1.0e-3
        )
        Random.seed!(1337)
        @test schmidt_number(random_state(Float64, 6), 2, [2, 3], 1; solver = SCS.Optimizer) ≤ 0
    end
    @testset "GME entanglement" begin
        for R ∈ (Float64, Double64)
            ρ = state_ghz(R, 2, 3)

            v, W = ppt_mixture(ρ, [2, 2, 2])
            @test isapprox(v, 0.4285, atol = 1.0e-3)
            full_body_basis = collect(Iterators.flatten(n_body_basis(i, 3) for i ∈ 0:3))
            v, w = ppt_mixture(ρ, [2, 2, 2], full_body_basis)
            @test isapprox(v, 0.4285, atol = 1.0e-3)
            @test isapprox(sum(w[i] * full_body_basis[i] for i ∈ eachindex(w)), W, atol = 1.0e-3)

            two_body_basis = collect(Iterators.flatten(n_body_basis(i, 3) for i ∈ 0:2))
            v, w = ppt_mixture(state_w(3), [2, 2, 2], two_body_basis)
            @test isapprox(v, 0.696, atol = 1.0e-3)
        end
    end
end
