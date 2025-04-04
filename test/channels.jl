@testset "Channels              " begin
    for R ∈ (Float64, Double64), T ∈ (R, Complex{R}) #Float128 and BigFloat take too long
        ρ = random_state(T, 2)
        p = R(7) / 10
        γ = R(8) / 10
        @test applykraus(channel_bit_flip(p), ρ) ≈ Hermitian(
            [ρ[1, 1]*p+ρ[2, 2]*(1-p) ρ[1, 2]*p+ρ[2, 1]*(1-p); ρ[2, 1]*p+ρ[1, 2]*(1-p) ρ[1, 1]*(1-p)+ρ[2, 2]*p]
        )
        @test applykraus(channel_phase_flip(p), ρ) ≈ Hermitian([ρ[1, 1] ρ[1, 2]*(2p-1); ρ[2, 1]*(2p-1) ρ[2, 2]])
        @test applykraus(channel_bit_phase_flip(p), ρ) ≈ Hermitian(
            [ρ[1, 1]*p+ρ[2, 2]*(1-p) ρ[1, 2]*p-ρ[2, 1]*(1-p); ρ[2, 1]*p-ρ[1, 2]*(1-p) ρ[1, 1]*(1-p)+ρ[2, 2]*p]
        )
        @test applykraus(channel_amplitude_damping(γ), ρ) ≈
              Hermitian([ρ[1, 1]+ρ[2, 2]*γ ρ[1, 2]*sqrt(1 - γ); ρ[2, 1]*sqrt(1 - γ) ρ[2, 2]*(1-γ)])
        ρ_st = ρ
        for _ ∈ 1:50
            ρ_st = applykraus(channel_amplitude_damping_generalized(p, γ), ρ_st)
        end
        @test ρ_st ≈ Hermitian([p 0; 0 1-p])
        @test applykraus(channel_phase_damping(γ), ρ) ≈  applykraus(channel_phase_flip((1 + sqrt(1 − γ)) / 2), ρ)
        ρ3 = random_state(T, 3)
        @test applykraus(channel_depolarizing(p, 3), ρ3) ≈ white_noise(ρ3, p)
        din, dout = 2, 3
        K = [randn(T, dout, din) for _ ∈ 1:3]
        @test diamond_norm(K) ≈ diamond_norm(choi(K), [din, dout]) atol = 1.0e-8 rtol = sqrt(_rtol(T))
    end
end
