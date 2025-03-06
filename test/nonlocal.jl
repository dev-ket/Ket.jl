@testset "Nonlocal games        " begin
    @test eltype(chsh()) <: Float64
    @test eltype(cglmp()) <: Float64
    @test eltype(inn22()) <: Int
    for T ∈ [Float64, Double64, Float128, BigFloat]
        @test eltype(chsh(T)) <: T
        @test eltype(cglmp(T)) <: T
        @test cglmp(T, 4)[3] == T(1) / 12
    end
end

@testset "Local bound           " begin
    @test local_bound(chsh()) ≈ 0.75
    @test local_bound(chsh(Int, 3)) == 6
    @test local_bound(cglmp(Int, 4)) == 9
    @test local_bound(gyni(Int, 3)) == 1
    @test local_bound(gyni(Int, 4)) == 1
    @test local_bound(braunsteincaves(Int)) == 5
    @test local_bound([1 2; 2 -2]; marg = false) == 5
    Random.seed!(0)
    fp1 = rand(0:1, 2, 2, 2, 2, 2, 2, 2, 2)
    fc1 = tensor_correlation(fp1)
    @test local_bound(fc1; correlation = true) ≈ local_bound(fp1)
    @test local_bound(fp1) == 12
    for T ∈ [Float64, Double64, Float128, BigFloat]
        fp1 = randn(T, 2, 2, 3, 4)
        fp2 = permutedims(fp1, (2, 1, 4, 3))
        fc1 = tensor_correlation(fp1)
        fc2 = tensor_correlation(fp2)
        @test local_bound(fp1) ≈ local_bound(fp2)
        @test local_bound(fc1) ≈ local_bound(fc2)
        @test local_bound(fc1) ≈ local_bound(fp1)
        bigfc1 = zeros(T, 5, 6)
        @views bigfc1[2:5, 2:6] .= fc1
        @test local_bound(fc1; marg = false) ≈ local_bound(bigfc1; marg = true)
        fp1 = rand(T, 2, 2, 2, 3, 4, 5)
        fp2 = permutedims(fp1, (3, 2, 1, 6, 5, 4))
        fc1 = tensor_correlation(fp1)
        fc2 = tensor_correlation(fp2)
        @test local_bound(fp1) ≈ local_bound(fp2)
        @test local_bound(fc1) ≈ local_bound(fc2)
        @test local_bound(fc1) ≈ local_bound(fp1)
        bigfc1 = zeros(T, 5, 6, 7)
        @views bigfc1[2:5, 2:6, 2:7] .= fc1
        @test local_bound(fc1; marg = false) ≈ local_bound(bigfc1; marg = true)
    end
end

@testset "Tsirelson bound       " begin
    @test tsirelson_bound(inn22(), (2, 2, 3, 3), 1)[1] ≈ 11 / 8
    chsh_fc = [
        0 0 0
        0 1 1
        0 1 -1
    ]
    @test all(tsirelson_bound(chsh_fc, 1) .≈ (2√2, [1 0 0; 0 1/√2 1/√2; 0 1/√2 -1/√2]))
    @test all(tsirelson_bound(chsh_fc, "1 + A B") .≈ (2√2, [1 0 0; 0 1/√2 1/√2; 0 1/√2 -1/√2]))
    @test all(tsirelson_bound(chsh_fc, 2) .≈ (2√2, [1 0 0; 0 1/√2 1/√2; 0 1/√2 -1/√2]))
    τ = Double64(9) / 10
    tilted_chsh_fc = [
        0 τ 0
        τ 1 1
        0 1 -1
    ]
    @test tsirelson_bound(tilted_chsh_fc, 3)[1] ≈ 3.80128907501837942169727948014219026
    bc_cg = tensor_collinsgisin(braunsteincaves())
    @test tsirelson_bound(bc_cg, (2, 2, 3, 3), 1)[1] ≈ cos(π / 12)^2 rtol = 1e-7
    @test tsirelson_bound(bc_cg, (2, 2, 3, 3), "1 + A B")[1] ≈ cos(π / 12)^2 rtol = 1e-7
    scenario = (2, 3, 3, 2)
    rand_cg = tensor_collinsgisin(randn(scenario))
    q, behaviour = tsirelson_bound(rand_cg, scenario, "1 + A B")
    @test q ≈ dot(rand_cg, behaviour)
    cglmp_cg = tensor_collinsgisin(cglmp())
    @test tsirelson_bound(cglmp_cg, (3, 3, 2, 2), "1 + A B")[1] ≈ (15 + sqrt(33)) / 24 rtol = 1.0e-7
    gyni_cg = tensor_collinsgisin(gyni())
    @test tsirelson_bound(gyni_cg, (2, 2, 2, 2, 2, 2), 3)[1] ≈ 0.25 rtol = 1e-6
    Śliwa18 = [
        0 0 0; 1 1 0; 1 1 0;;;
        0 -2 0; 1 0 1; 1 0 -1;;;
        0 0 2; 0 1 -1; 0 -1 -1
    ]
    @test tsirelson_bound(Śliwa18, 2)[1] ≈ 2 * (7 - sqrt(17)) rtol = 1e-7
end

@testset "Seesaw                " begin
    Random.seed!(1337)
    cglmp_cg = tensor_collinsgisin(cglmp())
    @test seesaw(cglmp_cg, (3, 3, 2, 2), 3)[1] ≈ (15 + sqrt(33)) / 24
    @test seesaw(inn22(), (2, 2, 3, 3), 2)[1] ≈ 1.25
end

@testset "FP and FC notations   " begin
    for T ∈ [Float64, Double64, Float128, BigFloat]
        Aax = povm(mub(Complex{T}, 2))
        fc_phiplus = Diagonal([1, 1, 1, -1])
        @test tensor_correlation(state_phiplus(Complex{T}), Aax, 2) ≈ fc_phiplus
        @test tensor_correlation(state_phiplus(Complex{T}), Aax, 2; marg = false) ≈ fc_phiplus[2:end, 2:end]
        fc_ghz = zeros(Int, 4, 4, 4)
        fc_ghz[[1, 6, 18, 21, 43, 48, 60, 63]] .= [1, 1, 1, 1, 1, -1, -1, -1]
        @test tensor_correlation(state_ghz(Complex{T}), Aax, 3) ≈ fc_ghz
        @test tensor_correlation(state_ghz(Complex{T}), Aax, 3; marg = false) ≈ fc_ghz[2:end, 2:end, 2:end]
        scenario = (2, 2, 2, 2, 3, 4)
        p = randn(T, scenario)
        mfc = randn(T, scenario[4:6] .+ 1)
        @test dot(mfc, tensor_correlation(p, true)) ≈ dot(tensor_probability(mfc, false), p)
        pfc = mfc
        m = p
        @test dot(tensor_correlation(m, false), pfc) ≈ dot(m, tensor_probability(pfc, true))
    end
end

@testset "FP and CG notations   " begin
    for T ∈ [Float64, Double64, Float128, BigFloat]
        Aax = povm(mub(Complex{T}, 2))
        cg_phiplus = [1.0 0.5 0.5 0.5; 0.5 0.5 0.25 0.25; 0.5 0.25 0.5 0.25; 0.5 0.25 0.25 0.0]
        @test tensor_collinsgisin(state_phiplus(Complex{T}), Aax, 2) ≈ cg_phiplus
        scenario = (2, 3, 4, 5, 6, 7)
        mcg = randn(T, scenario[4:6] .* (scenario[1:3] .- 1) .+ 1)
        p = randn(T, scenario)
        @test dot(mcg, tensor_collinsgisin(p, true)) ≈ dot(tensor_probability(mcg, scenario, false), p)
        pcg = mcg
        m = p
        @test dot(tensor_collinsgisin(m, false), pcg) ≈ dot(m, tensor_probability(pcg, scenario, true))
        @test tensor_collinsgisin(tensor_probability(pcg, scenario, true), true) ≈ pcg
    end
end

@testset "CG and FC notations   " begin
    for T ∈ [Float64, Double64, Float128, BigFloat]
        scenario = (2, 2, 2, 2, 3, 4)
        matrix_size = scenario[4:6] .* (scenario[1:3] .- 1) .+ 1
        mfc = randn(T, matrix_size)
        pcg = randn(T, matrix_size)
        @test dot(mfc, tensor_correlation(pcg, true; collinsgisin = true)) ≈
              dot(tensor_collinsgisin(mfc; correlation = true), pcg)
        pfc = mfc
        mcg = pcg
        @test dot(tensor_correlation(mcg; collinsgisin = true), pfc) ≈
              dot(mcg, tensor_collinsgisin(pfc, true; correlation = true))
        pcg = tensor_collinsgisin(tensor_probability(pfc, true), true)
        @test tensor_collinsgisin(pfc, true; correlation = true) ≈ pcg
        @test tensor_correlation(pcg, true; collinsgisin = true) ≈ pfc
        @test tensor_correlation(pcg, true; collinsgisin = true, marg = false) ≈
              tensor_correlation(tensor_probability(pcg, scenario, true), true; marg = false)
        mfc = pfc
        mcg = tensor_collinsgisin(tensor_probability(mfc))
        @test tensor_collinsgisin(mfc; correlation = true) ≈ mcg
        @test tensor_correlation(mcg; collinsgisin = true) ≈ mfc
        @test tensor_correlation(mcg; collinsgisin = true, marg = false) ≈
              tensor_correlation(tensor_probability(mcg, scenario); marg = false)
    end
end

@testset "Nonlocality robustness" begin
    for T ∈ [Float64, Double64]
        prbox = 2 * chsh(T)
        @test nonlocality_robustness(prbox; noise = "white") ≈ T(1)
        @test nonlocality_robustness(prbox; noise = "local") ≈ T(1) / 2
        @test nonlocality_robustness(prbox; noise = "general") ≈ T(1) / 3
    end
end
