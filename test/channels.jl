@testset "Channels              " begin
    for R ∈ (Float64, Double64), T ∈ (R, Complex{R}) #BigFloat takes too long
        ρ = random_state(T, 2)
        p = R(7) / 10
        γ = R(8) / 10
        @test applymap(channel_bit_flip(p), ρ) ≈ Hermitian(
            [ρ[1, 1]*p+ρ[2, 2]*(1-p) ρ[1, 2]*p+ρ[2, 1]*(1-p); ρ[2, 1]*p+ρ[1, 2]*(1-p) ρ[1, 1]*(1-p)+ρ[2, 2]*p]
        )
        @test applymap(channel_phase_flip(p), ρ) ≈ Hermitian([ρ[1, 1] ρ[1, 2]*(2p-1); ρ[2, 1]*(2p-1) ρ[2, 2]])
        @test applymap(channel_bit_phase_flip(p), ρ) ≈ Hermitian(
            [ρ[1, 1]*p+ρ[2, 2]*(1-p) ρ[1, 2]*p-ρ[2, 1]*(1-p); ρ[2, 1]*p-ρ[1, 2]*(1-p) ρ[1, 1]*(1-p)+ρ[2, 2]*p]
        )
        K = channel_amplitude_damping(γ)
        @test applymap(K, ρ) ≈ Hermitian([ρ[1, 1]+ρ[2, 2]*γ ρ[1, 2]*sqrt(1 - γ); ρ[2, 1]*sqrt(1 - γ) ρ[2, 2]*(1-γ)])
        ρ_st = applymap(channel_amplitude_damping_generalized(p, γ), ρ)
        X = [0 1; 1 0]
        @test ρ_st ≈ p * applymap(K, ρ) + (1 - p) * X * applymap(K, X * ρ * X) * X
        @test applymap(channel_phase_damping(γ), ρ) ≈ applymap(channel_phase_flip((1 + sqrt(1 − γ)) / 2), ρ)
        ρ3 = random_state(T, 3)
        @test applymap(channel_depolarizing(p, 3), ρ3) ≈ white_noise(ρ3, p)
        din, dout = 2, 3
        K = [randn(T, dout, din) for _ ∈ 1:3]
        Φ = choi(K)
        @test applymap(K, ρ) ≈ applymap(Φ, ρ)
        @test diamond_norm(K) ≈ diamond_norm(Φ, [din, dout]) atol = 1.0e-8 rtol = sqrt(_rtol(T))
    end
    model = JuMP.Model()
    JuMP.@variable(model, ρ[1:2, 1:2], Hermitian)
    JuMP.@variable(model, ρreal[1:2, 1:2], Symmetric)
    K = [randn(ComplexF64, 3, 2) for _ ∈ 1:2]
    Kreal = real(K)
    Φ = choi(K)
    Φreal = choi(Kreal)
    @test isa(applymap(K, ρ), Hermitian)
    @test isa(applymap(Φ, ρ), Hermitian)
    @test isa(applymap(K, ρreal), Hermitian)
    @test isa(applymap(Φ, ρreal), Hermitian)
    @test isa(applymap(Kreal, ρ), Hermitian)
    @test isa(applymap(Φreal, ρ), Hermitian)
    @test isa(applymap(Kreal, ρreal), Symmetric)
    @test isa(applymap(Φreal, ρreal), Symmetric)
    JuMP.@variable(model, Φ[1:6, 1:6], Hermitian)
    JuMP.@variable(model, Φreal[1:6, 1:6], Symmetric)
    ρ = Hermitian(randn(ComplexF64, 2, 2))
    ρreal = Symmetric(randn(2, 2))
    @test isa(applymap(Φ, ρ), Hermitian)
    @test isa(applymap(Φ, ρreal), Hermitian)
    @test isa(applymap(Φreal, ρ), Hermitian)
    @test isa(applymap(Φreal, ρreal), Symmetric)
    sK = [sprandn(ComplexF64, 3, 2, 0.5) for _ ∈ 1:2]
    sΦ = choi(sK)
    sρ = Hermitian(sprandn(ComplexF64, 2, 2, 0.5))
    @test issparse(applymap(sK, sρ))
    @test issparse(applymap(sΦ, sρ))
end
