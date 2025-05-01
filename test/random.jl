@testset "Random                " begin
    @testset "States" begin
        ρ = random_state(3)
        @test isa(ρ, Hermitian{ComplexF64})
        ψ = random_state_ket(3)
        @test isa(ψ, Vector{ComplexF64})
        for R ∈ (Float64, BigFloat), T ∈ (R, Complex{R})
            Random.seed!(1337) #make ranks behave
            ψ = random_state_ket(T, 3)
            @test ψ' * ψ ≈ 1
            @test isa(ψ, Vector{T})
            ρ = random_state(T, 3)
            @test tr(ρ) ≈ 1
            @test rank(ρ; rtol = _rtol(T)) == 3
            ρ = random_state(T, 3, 2)
            @test tr(ρ) ≈ 1
            @test rank(ρ; rtol = _rtol(T)) == 2
            ρ = random_state(T, 3, 1)
            @test tr(ρ) ≈ 1
            @test rank(ρ; rtol = _rtol(T)) == 1
            @test minimum(eigvals(ρ)) > -_rtol(T)
            @test isa(ρ, Hermitian{T})
        end
    end
    @testset "Unitaries" begin
        U = random_unitary(3)
        V = random_isometry(3, 2)
        @test eltype(U) <: ComplexF64
        @test eltype(V) <: ComplexF64
        for R ∈ (Float64, Double64, BigFloat), T ∈ (R, Complex{R})
            U = random_unitary(T, 3)
            V = random_isometry(T, 3, 2)
            @test U * U' ≈ I(3)
            @test V' * V ≈ I(2)
        end
    end
    @testset "Probability" begin
        p = random_probability(3)
        @test isa(p, Vector{Float64})
        for T ∈ (Float64, BigFloat)
            p = random_probability(T, 5)
            @test sum(p) ≈ 1
            @test minimum(p) ≥ 0
            @test isa(p, Vector{T})
        end
    end
    @testset "POVM" begin
        for R ∈ (Float64, BigFloat), T ∈ (R, Complex{R})
            E = random_povm(T, 2, 3)
            @test test_povm(E)
            for i ∈ 1:length(E)
                @test rank(E[i]; rtol = _rtol(T)) == 2
            end
            E = random_povm(T, 2, 3, 1)
            @test test_povm(E)
            for i ∈ 1:length(E)
                @test rank(E[i]; rtol = _rtol(T)) == 1
            end
        end
    end
end
