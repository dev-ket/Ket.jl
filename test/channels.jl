@testset "Channels              " begin
    for R ∈ (Float64, Float64x2), T ∈ (R, Complex{R})
        ρ = random_state(T, 2)
        p = R(7) / 10
        γ = R(8) / 10
        @test applymap(ρ, channel_bit_flip(p)) ≈ Hermitian(
            [ρ[1, 1]*p+ρ[2, 2]*(1-p) ρ[1, 2]*p+ρ[2, 1]*(1-p); ρ[2, 1]*p+ρ[1, 2]*(1-p) ρ[1, 1]*(1-p)+ρ[2, 2]*p]
        )
        @test applymap(ρ, channel_phase_flip(p)) ≈ Hermitian([ρ[1, 1] ρ[1, 2]*(2p-1); ρ[2, 1]*(2p-1) ρ[2, 2]])
        @test applymap(ρ, channel_bit_phase_flip(p)) ≈ Hermitian(
            [ρ[1, 1]*p+ρ[2, 2]*(1-p) ρ[1, 2]*p-ρ[2, 1]*(1-p); ρ[2, 1]*p-ρ[1, 2]*(1-p) ρ[1, 1]*(1-p)+ρ[2, 2]*p]
        )
        K = channel_amplitude_damping(γ)
        @test applymap(ρ, K) ≈ Hermitian([ρ[1, 1]+ρ[2, 2]*γ ρ[1, 2]*sqrt(1 - γ); ρ[2, 1]*sqrt(1 - γ) ρ[2, 2]*(1-γ)])
        ρ_st = applymap(ρ, channel_amplitude_damping_generalized(p, γ))
        X = [0 1; 1 0]
        @test ρ_st ≈ p * applymap(ρ, K) + (1 - p) * X * applymap(X * ρ * X, K) * X
        @test applymap(ρ, channel_phase_damping(γ)) ≈ applymap(ρ, channel_phase_flip((1 + sqrt(1 − γ)) / 2))
        ρ3 = random_state(T, 3)
        K = channel_depolarizing(p, 3)
        ρ3_depolarized = applymap(ρ3, K)
        @test ρ3_depolarized ≈ applymap_depolarizing(ρ3, p)
        @test applymap_depolarizing(ρ3, p) ≈ ρ3_depolarized
        @test isa(applymap_depolarizing(ρ3, p), Hermitian)
        result = zeros(T, 3, 3)
        @test applymap_depolarizing!(result, ρ3, p) === result
        @test result ≈ ρ3_depolarized
        ρ3_inplace = Matrix(ρ3)
        @test applymap_depolarizing!(ρ3_inplace, p) === ρ3_inplace
        @test ρ3_inplace ≈ ρ3_depolarized

        M3 = randn(T, 3, 3)
        @test applymap_depolarizing(M3, p) ≈ applymap(M3, K)

        K = channel_loss(p, 3)
        ρ3_loss = applymap(ρ3, K)
        @test ρ3_loss ≈ Hermitian(cat(p .* ρ3, (1 - p) * tr(ρ3), dims=[1, 2]))
        @test applymap_loss(ρ3, p) ≈ ρ3_loss
        @test isa(applymap_loss(ρ3, p), Hermitian)
        result = zeros(T, 4, 4)
        @test applymap_loss!(result, ρ3, p) === result
        @test result ≈ ρ3_loss
        @test applymap_loss(M3, p) ≈ applymap(M3, K)
        din, dout = 2, 3
        K = [randn(T, dout, din) for _ ∈ 1:3]
        Kd = [Ki' for Ki ∈ K]
        @test applymap(ρ, K) ≈ applymap(ρ, Kd; dual=true)
        Φ = choi(K)
        Φd = choi(Kd)
        @test applymap(ρ, Φ) ≈ applymap(ρ, Φd; dual=true)
        @test applymap(ρ, K) ≈ applymap(ρ, Φ)
        @test diamond_norm(K) ≈ diamond_norm(Φ, [din, dout]) atol = 1.0e-8 rtol = sqrt(_rtol(T))
        Λ(x) = sum(Ki * x * Ki' for Ki ∈ K)
        @test Φ ≈ choi(Λ, din)
    end
    @test choi(tr, 2) == I(2)
    @test choi(transpose, 2) == [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
    model = JuMP.Model()
    JuMP.@variable(model, ρ[1:2, 1:2], Hermitian)
    JuMP.@variable(model, ρreal[1:2, 1:2], Symmetric)
    K = [randn(ComplexF64, 3, 2) for _ ∈ 1:2]
    Kreal = real(K)
    Φ = choi(K)
    Φreal = choi(Kreal)
    @test isa(applymap(ρ, K), Hermitian)
    @test isa(applymap(ρ, Φ), Hermitian)
    @test isa(applymap(ρreal, K), Hermitian)
    @test isa(applymap(ρreal, Φ), Hermitian)
    @test isa(applymap(ρ, Kreal), Hermitian)
    @test isa(applymap(ρ, Φreal), Hermitian)
    @test isa(applymap(ρreal, Kreal), Symmetric)
    @test isa(applymap(ρreal, Φreal), Symmetric)
    @test isa(applymap_depolarizing(ρ, 0.7), Hermitian)
    @test isa(applymap_loss(ρ, 0.7), Hermitian)
    @test isa(applymap_depolarizing(ρreal, 0.7), Symmetric)
    @test isa(applymap_loss(ρreal, 0.7), Symmetric)
    JuMP.@variable(model, Φ[1:6, 1:6], Hermitian)
    JuMP.@variable(model, Φreal[1:6, 1:6], Symmetric)
    ρ = Hermitian(randn(ComplexF64, 2, 2))
    ρreal = Symmetric(randn(2, 2))
    @test isa(applymap(ρ, Φ), Hermitian)
    @test isa(applymap(ρreal, Φ), Hermitian)
    @test isa(applymap(ρ, Φreal), Hermitian)
    @test isa(applymap(ρreal, Φreal), Symmetric)
    sK = [sprandn(ComplexF64, 3, 2, 0.5) for _ ∈ 1:2]
    sΦ = choi(sK)
    sρ = sprandn(ComplexF64, 2, 2, 0.8)
    sρ += sρ'
    sρ = Hermitian(sρ)
    @test applymap(sρ, sK) ≈ applymap(sρ, sΦ)
    @test issparse(applymap(sρ, sK))
    @test issparse(applymap(sρ, sΦ))
    p = 0.7
    @test applymap_depolarizing(sρ, p) ≈ applymap(sρ, channel_depolarizing(p, 2))
    @test applymap_loss(sρ, p) ≈ applymap(sρ, channel_loss(p, 2))
    @test issparse(applymap_depolarizing(sρ, p))
    @test issparse(applymap_loss(sρ, p))
end
