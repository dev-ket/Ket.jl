@testset "States                " begin
    for R ∈ [Float64, BigFloat]
        T = Complex{R}
        ψ = state_phiplus_ket(T)
        @test ψ == inv(sqrt(R(2))) * (ket(1, 4) + ket(4, 4))
        @test ketbra(ψ) ≈ state_phiplus(T)
        v = R(8)/10
        @test v*ketbra(ψ) + (1-v)*I/4 ≈ state_phiplus(T; v)
        ψ = state_psiminus_ket(T)
        @test ψ == inv(sqrt(R(2))) * (ket(2, 4) - ket(3, 4))
        @test ketbra(ψ) ≈ state_psiminus(T)
        ψ = state_w_ket(T)
        @test ψ == inv(sqrt(R(3))) * (ket(2, 8) + ket(3, 8) + ket(5, 8))
        @test ketbra(ψ) ≈ state_w(T)
        coeff = T.([1, 2, 2]) / 9
        @test state_w_ket(T; coeff) == T.(ket(2, 8) + 2 * ket(3, 8) + 2 * ket(5, 8)) / 9
        ψ = state_dicke_ket(T, 2, 3)
        @test ψ == inv(sqrt(R(3))) * (ket(4, 8) + ket(6, 8) + ket(7, 8))
        @test ketbra(ψ) ≈ state_dicke(T, 2, 3)
        ψ = state_ghz_ket(T)
        @test ψ == inv(sqrt(R(2))) * (ket(1, 8) + ket(8, 8))
        @test ketbra(ψ) ≈ state_ghz(T)
        coeff = [T(3) / 5, T(4) / 5]
        @test state_ghz_ket(T; coeff) == (T(3) * ket(1, 8) + T(4) * ket(8, 8)) / 5
        @test state_supersinglet(T, 2) ≈ state_psiminus(T)
        U = foldl(kron, fill(Matrix(random_unitary(T, 3)), 3))
        rho = state_supersinglet(T, 3)
        @test U * rho * U' ≈ rho
        @test kron(I(5),shift(5,3)*clock(5,2))*state_phiplus_ket(5) ≈ state_bell_ket(3,2,5)
        rho = state_horodecki33(T, 1)
        ref = Matrix{T}(I, 9, 9)
        ref[1:4:9, 1:4:9] .= 1
        ref ./= 9
        @test rho ≈ ref
        @test minimum(eigvals(partial_transpose(rho, 1))) ≥ 0
        rho = state_horodecki24(T, 1)
        ref = Matrix{T}(I, 8, 8)
        ref[1:5:8, 1:5:8] .= 1
        ref[2:5:8, 2:5:8] .= 1
        ref[3:5:8, 3:5:8] .= 1
        ref ./= 8
        @test rho ≈ ref
        @test minimum(eigvals(partial_transpose(rho, 1, [2, 4]))) ≥ 0
        edges = [[(1, 1)], [(1, 2)], [(2, 1)], [(3, 4)], [(3, 4)], 
            [(4, 3)], [(4, 3)], [(2, 5)], [(5, 2)], [(3, 2),(5, 4)],
            [(3, 3),(4, 4)], [(2, 3),(4, 5)], [(1, 3),(2, 2),(3, 1)]]
        @test minimum(eigvals(partial_transpose(state_grid(T, edges,5, 5), 1, [5, 5]))) ≥ 0 
        @test minimum(eigvals(partial_transpose(state_crosshatch(T), 1, [3, 3]))) ≥ 0 
        ψ = state_sindici_piani_ket(T, 4)
        @test ketbra(ψ) ≈ state_sindici_piani(T, 4)
    end
end
