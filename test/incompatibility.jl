@testset "Incompatibility       " begin
    A = povm(mub(2))
    @test incompatibility_robustness(A; noise = :depolarizing) ≈ √3 - 1 rtol = 1e-7
    @test incompatibility_robustness(A; noise = :random) ≈ √3 - 1 rtol = 1e-7
    @test incompatibility_robustness(A; noise = :probabilistic) ≈ √3 - 1 rtol = 1e-7
    @test incompatibility_robustness(A; noise = :jointly_measurable) ≈ (√3 - 1) / 2 rtol = 1e-7
    @test incompatibility_robustness(A; noise = :general) ≈ 2 - √3 rtol = 1e-7
    # other types, also checks that the default noise is :general
    @test incompatibility_robustness(povm(broadcast.(Float64, mub(2, 2)))) ≈ 3 - 2√2 rtol = 1e-7
    @test incompatibility_robustness(povm((mub(Complex{Float64x2}, 2)))) ≈ 2 - √Float64x2(3) atol = 2e-13
end
