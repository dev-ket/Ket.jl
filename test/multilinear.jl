@testset "Multilinear algebra   " begin
    @testset "Partial trace      " begin
        model = JuMP.Model()
        JuMP.@variable(model, ρ[1:4, 1:4], Symmetric)
        ptrace = [tr(ρ[1:2, 1:2]) tr(ρ[1:2, 3:4]); tr(ρ[3:4, 1:2]) tr(ρ[3:4, 3:4])]
        @test partial_trace(ρ, 2, [2, 2]) == ptrace
        JuMP.@variable(model, σ[1:4, 1:4], Hermitian)
        ptrace = [tr(σ[1:2, 1:2]) tr(σ[1:2, 3:4]); tr(σ[3:4, 1:2]) tr(σ[3:4, 3:4])]
        @test partial_trace(σ, 2, [2, 2]) == ptrace
        d1, d2, d3 = 2, 2, 3
        for R ∈ (Float64, Double64, Float128, BigFloat), T ∈ (R, Complex{R})
            a = randn(T, d1, d1)
            b = randn(T, d2, d2)
            c = randn(T, d3, d3)
            ab = kron(a, b)
            ac = kron(a, c)
            bc = kron(b, c)
            abc = kron(ab, c)
            @test partial_trace(ab, [1, 2])[1] ≈ tr(ab)
            @test partial_trace(ab, 2) ≈ a * tr(b)
            @test partial_trace(ab, 1) ≈ b * tr(a)
            @test partial_trace(ab, Int[]) ≈ ab
            @test partial_trace(abc, [1, 2, 3], [d1, d2, d3])[1] ≈ tr(abc)
            @test partial_trace(abc, [2, 3], [d1, d2, d3]) ≈ a * tr(b) * tr(c)
            @test partial_trace(abc, [1, 3], [d1, d2, d3]) ≈ b * tr(a) * tr(c)
            @test partial_trace(abc, [1, 2], [d1, d2, d3]) ≈ c * tr(a) * tr(b)
            @test partial_trace(abc, 3, [d1, d2, d3]) ≈ ab * tr(c)
            @test partial_trace(abc, 2, [d1, d2, d3]) ≈ ac * tr(b)
            @test partial_trace(abc, 1, [d1, d2, d3]) ≈ bc * tr(a)
            @test partial_trace(abc, Int[], [d1, d2, d3]) ≈ abc
        end
        for wrapper ∈ (Symmetric, Hermitian)
            M = wrapper(randn(ComplexF64, (d1 * d2 * d3, d1 * d2 * d3)))
            x = Matrix(M)
            @test partial_trace(M, 2, [d1, d2, d3]) ≈ partial_trace(x, 2, [d1, d2, d3])
            @test partial_trace(M, [1, 3], [d1, d2, d3]) ≈ partial_trace(x, [1, 3], [d1, d2, d3])
        end
    end

    @testset "Partial transpose  " begin
        model = JuMP.Model()
        JuMP.@variable(model, ρ[1:4, 1:4], Symmetric)
        ptrans = [transpose(ρ[1:2, 1:2]) transpose(ρ[1:2, 3:4]); transpose(ρ[3:4, 1:2]) transpose(ρ[3:4, 3:4])]
        @test partial_transpose(ρ, 2, [2, 2]) == ptrans
        JuMP.@variable(model, σ[1:4, 1:4], Hermitian)
        ptrans = [transpose(σ[1:2, 1:2]) transpose(σ[1:2, 3:4]); transpose(σ[3:4, 1:2]) transpose(σ[3:4, 3:4])]
        @test partial_transpose(σ, 2, [2, 2]) == ptrans
        d1, d2, d3 = 2, 2, 3
        for R ∈ (Float64, Double64, Float128, BigFloat), T ∈ (R, Complex{R})
            a = randn(T, d1, d1)
            b = randn(T, d2, d2)
            c = randn(T, d3, d3)
            ab = kron(a, b)
            ac = kron(a, c)
            bc = kron(b, c)
            abc = kron(ab, c)
            @test partial_transpose(ab, [1, 2]) ≈ transpose(ab)
            @test partial_transpose(ab, 2) ≈ kron(a, transpose(b))
            @test partial_transpose(ab, 1) ≈ kron(transpose(a), b)
            @test partial_transpose(ab, Int[]) ≈ ab
            @test partial_transpose(abc, [1, 2, 3], [d1, d2, d3]) ≈ transpose(abc)
            @test partial_transpose(abc, [2, 3], [d1, d2, d3]) ≈ kron(a, transpose(b), transpose(c))
            @test partial_transpose(abc, [1, 3], [d1, d2, d3]) ≈ kron(transpose(a), b, transpose(c))
            @test partial_transpose(abc, [1, 2], [d1, d2, d3]) ≈ kron(transpose(a), transpose(b), c)
            @test partial_transpose(abc, 3, [d1, d2, d3]) ≈ kron(ab, transpose(c))
            @test partial_transpose(abc, 2, [d1, d2, d3]) ≈ kron(a, transpose(b), c)
            @test partial_transpose(abc, 1, [d1, d2, d3]) ≈ kron(transpose(a), bc)
            @test partial_transpose(abc, Int[], [d1, d2, d3]) ≈ abc
        end
        for wrapper ∈ (Symmetric, Hermitian)
            M = wrapper(randn(ComplexF64, (d1 * d2 * d3, d1 * d2 * d3)))
            x = Matrix(M)
            @test partial_transpose(M, 2, [d1, d2, d3]) ≈ partial_transpose(x, 2, [d1, d2, d3])
            @test partial_transpose(M, [1, 3], [d1, d2, d3]) ≈ partial_transpose(x, [1, 3], [d1, d2, d3])
        end
    end

    @testset "Permute systems    " begin
        @testset "Vectors" begin
            d1, d2, d3 = 2, 2, 3
            for R ∈ (Float64, Double64, Float128, BigFloat), T ∈ (R, Complex{R})
                u = randn(T, d1)
                v = randn(T, d2)
                w = randn(T, d3)
                uv = kron(u, v)
                vw = kron(v, w)
                uvw = kron(u, v, w)
                @test permute_systems(uv, [1, 2]) ≈ kron(u, v)
                @test permute_systems(uv, [2, 1]) ≈ kron(v, u)
                @test permute_systems(vw, [2, 1], [d2, d3]) ≈ kron(w, v)
                @test permute_systems(uvw, [1, 2, 3], [d1, d2, d3]) ≈ kron(u, v, w)
                @test permute_systems(uvw, [2, 3, 1], [d1, d2, d3]) ≈ kron(v, w, u)
                @test permute_systems(uvw, [3, 1, 2], [d1, d2, d3]) ≈ kron(w, u, v)
                @test permute_systems(uvw, [1, 3, 2], [d1, d2, d3]) ≈ kron(u, w, v)
                @test permute_systems(uvw, [2, 1, 3], [d1, d2, d3]) ≈ kron(v, u, w)
                @test permute_systems(uvw, [3, 2, 1], [d1, d2, d3]) ≈ kron(w, v, u)
            end
        end

        @testset "Square matrices" begin
            d1, d2, d3 = 2, 2, 3
            for R ∈ (Float64, Double64, Float128, BigFloat), T ∈ (R, Complex{R})
                a = randn(T, d1, d1)
                b = randn(T, d2, d2)
                c = randn(T, d3, d3)
                ab = kron(a, b)
                ac = kron(a, c)
                bc = kron(b, c)
                abc = kron(a, b, c)
                @test permute_systems(ab, [1, 2]) ≈ kron(a, b)
                @test permute_systems(ab, [2, 1]) ≈ kron(b, a)
                @test permute_systems(ac, [2, 1], [d1, d3]) ≈ kron(c, a)
                @test permute_systems(bc, [2, 1], [d2, d3]) ≈ kron(c, b)
                @test permute_systems(abc, [1, 2, 3], [d1, d2, d3]) ≈ kron(a, b, c)
                @test permute_systems(abc, [2, 3, 1], [d1, d2, d3]) ≈ kron(b, c, a)
                @test permute_systems(abc, [3, 1, 2], [d1, d2, d3]) ≈ kron(c, a, b)
                @test permute_systems(abc, [1, 3, 2], [d1, d2, d3]) ≈ kron(a, c, b)
                @test permute_systems(abc, [2, 1, 3], [d1, d2, d3]) ≈ kron(b, a, c)
                @test permute_systems(abc, [3, 2, 1], [d1, d2, d3]) ≈ kron(c, b, a)

                p = permutation_matrix(d1, [1, 2])
                @test permute_systems(ab, [1, 2]) ≈ p * ab * p'
                p = permutation_matrix([d1, d2], [2, 1])
                @test permute_systems(ab, [2, 1]) ≈ p * ab * p'
                p = permutation_matrix([d1, d2, d3], [1, 2, 3])
                @test permute_systems(abc, [1, 2, 3], [d1, d2, d3]) ≈ p * abc * p'
                p = permutation_matrix([d1, d2, d3], [2, 3, 1])
                @test permute_systems(abc, [2, 3, 1], [d1, d2, d3]) ≈ p * abc * p'
                p = permutation_matrix([d1, d2, d3], [3, 1, 2])
                @test permute_systems(abc, [3, 1, 2], [d1, d2, d3]) ≈ p * abc * p'
                p = permutation_matrix([d1, d2, d3], [1, 3, 2])
                @test permute_systems(abc, [1, 3, 2], [d1, d2, d3]) ≈ p * abc * p'
                p = permutation_matrix([d1, d2, d3], [2, 1, 3])
                @test permute_systems(abc, [2, 1, 3], [d1, d2, d3]) ≈ p * abc * p'
                p = permutation_matrix([d1, d2, d3], [3, 2, 1])
                @test permute_systems(abc, [3, 2, 1], [d1, d2, d3]) ≈ p * abc * p'
            end
            for wrapper ∈ (Symmetric, Hermitian)
                M = wrapper(randn(ComplexF64, (d1 * d2 * d3, d1 * d2 * d3)))
                x = Matrix(M)
                @test permute_systems(M, [3, 1, 2], [d1, d2, d3]) ≈ permute_systems(x, [3, 1, 2], [d1, d2, d3])
                @test permute_systems(M, [1, 3, 2], [d1, d2, d3]) ≈ permute_systems(x, [1, 3, 2], [d1, d2, d3])
            end
        end

        @testset "Rectangular matrices" begin
            d1, d2, d3 = 2, 3, 4
            for R ∈ (Float64, Double64, Float128, BigFloat), T ∈ (R, Complex{R})
                a = randn(T, d1, d2)
                b = randn(T, d1, d3)
                c = randn(T, d2, d3)
                ab = kron(a, b)
                ac = kron(a, c)
                bc = kron(b, c)
                abc = kron(a, b, c)
                @test permute_systems(ab, [1, 2], [d1 d2; d1 d3]) ≈ kron(a, b)
                @test permute_systems(ab, [2, 1], [d1 d2; d1 d3]) ≈ kron(b, a)
                @test permute_systems(ac, [2, 1], [d1 d2; d2 d3]) ≈ kron(c, a)
                @test permute_systems(bc, [2, 1], [d1 d3; d2 d3]) ≈ kron(c, b)
                @test permute_systems(abc, [1, 2, 3], [d1 d2; d1 d3; d2 d3]) ≈ kron(a, b, c)
                @test permute_systems(abc, [2, 3, 1], [d1 d2; d1 d3; d2 d3]) ≈ kron(b, c, a)
                @test permute_systems(abc, [3, 1, 2], [d1 d2; d1 d3; d2 d3]) ≈ kron(c, a, b)
                @test permute_systems(abc, [1, 3, 2], [d1 d2; d1 d3; d2 d3]) ≈ kron(a, c, b)
                @test permute_systems(abc, [2, 1, 3], [d1 d2; d1 d3; d2 d3]) ≈ kron(b, a, c)
                @test permute_systems(abc, [3, 2, 1], [d1 d2; d1 d3; d2 d3]) ≈ kron(c, b, a)
            end
        end
    end
    @testset "Trace replace      " begin
        model = JuMP.Model()
        JuMP.@variable(model, ρ[1:4, 1:4], Hermitian)
        trrp = [
            tr(ρ[1:2, 1:2])./2 0 tr(ρ[1:2, 3:4])./2 0
            0 tr(ρ[1:2, 1:2])./2 0 tr(ρ[1:2, 3:4])./2
            tr(ρ[3:4, 1:2])./2 0 tr(ρ[3:4, 3:4])./2 0
            0 tr(ρ[3:4, 1:2])./2 0 tr(ρ[3:4, 3:4])./2
        ]
        @test trace_replace(ρ, 2, [2, 2]) == trrp
        d1, d2, d3 = 2, 2, 3
        for R ∈ (Float64, Double64, Float128, BigFloat), T ∈ (R, Complex{R})
            a = randn(T, d1, d1)
            b = randn(T, d2, d2)
            c = randn(T, d3, d3)
            ab = kron(a, b)
            ac = kron(a, c)
            bc = kron(b, c)
            abc = kron(ab, c)
            I2 = Matrix(one(T) * I, (2, 2)) ./ 2 #Normalized identity
            I3 = Matrix(one(T) * I, (3, 3)) ./ 3
            I4 = Matrix(one(T) * I, (4, 4)) ./ 4
            I6 = Matrix(one(T) * I, (6, 6)) ./ 6
            I12 = Matrix(one(T) * I, (12, 12)) ./ 12
            @test trace_replace(ab, [1, 2]) ≈ tr(ab) * I4
            @test trace_replace(ab, 2) ≈ kron(partial_trace(ab, 2), I2)
            @test trace_replace(ab, 1) ≈ kron(I2, partial_trace(ab, 1))
            @test trace_replace(ab, Int[]) ≈ ab
            @test trace_replace(abc, [1, 2, 3], [d1, d2, d3]) ≈ I12 * tr(abc)
            @test trace_replace(abc, [2, 3], [d1, d2, d3]) ≈ kron(partial_trace(abc, [2, 3], [d1, d2, d3]), I6)
            @test trace_replace(abc, [1, 3], [d1, d2, d3]) ≈
                  permute_systems(kron(partial_trace(abc, [1, 3], [d1, d2, d3]), I6), [2, 1, 3], [d1, d2, d3])
            @test trace_replace(abc, [1, 2], [d1, d2, d3]) ≈ kron(I4, partial_trace(abc, [1, 2], [d1, d2, d3]))
            @test trace_replace(abc, 3, [d1, d2, d3]) ≈ kron(partial_trace(abc, 3, [d1, d2, d3]), I3)
            @test trace_replace(abc, 2, [d1, d2, d3]) ≈
                  permute_systems(kron(partial_trace(abc, 2, [d1, d2, d3]), I2), [1, 3, 2], [d1, d3, d2])
            @test trace_replace(abc, 1, [d1, d2, d3]) ≈ kron(I2, partial_trace(abc, 1, [d1, d2, d3]))
            @test trace_replace(abc, Int[], [d1, d2, d3]) ≈ abc
        end
        for wrapper ∈ (Symmetric, Hermitian)
            M = wrapper(randn(ComplexF64, (d1 * d2 * d3, d1 * d2 * d3)))
            x = Matrix(M)
            @test trace_replace(M, 2, [d1, d2, d3]) ≈ trace_replace(x, 2, [d1, d2, d3])
            @test trace_replace(M, [1, 3], [d1, d2, d3]) ≈ trace_replace(x, [1, 3], [d1, d2, d3])
        end
    end
end

#TODO add test with JuMP variables
