@testset "Parameterizations     " begin
    d = 5
    for R ∈ (Float64, Float64x2)
        λ = randn(R, d, d)
        U = parameterized_unitary(λ)
        @test U * U' ≈ I
    end
end
