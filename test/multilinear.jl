@testset "Multilinear algebra   " begin
    @testset "Apply to subsystem      " begin
        @testset "Square op applied to ψ " begin
            model = JuMP.Model()
            H = [1 1; 1 -1]
            JuMP.@variable(model, ψ[1:4])
            res = Array{JuMP.AbstractJuMPScalar}(undef, 4)
            res[1:2] .= H * ψ[1:2]
            res[3:4] .= H * ψ[3:4]
            @test apply_to_subsystem(H, ψ, 2, [2, 2]) == res
            for R ∈ (Float64, BigFloat), T ∈ (R, Complex{R})
                a = randn(T, 2, 2)
                b = randn(T, 2, 2)
                c = randn(T, 3, 3)
                v1 = randn(T, 2)
                v2 = randn(T, 2)
                v3 = randn(T, 3)
                ab = kron(a, b)
                ac = kron(a, c)
                bc = kron(b, c)
                abc = kron(ab, c)
                v12 = kron(v1, v2)
                v13 = kron(v1, v3)
                v23 = kron(v2, v3)
                v123 = kron(v1, v2, v3)
                I2 = Matrix(one(T) * I, (2, 2))
                I3 = Matrix(one(T) * I, (3, 3))
                I4 = Matrix(one(T) * I, (4, 4))
                I6 = Matrix(one(T) * I, (6, 6))
                @test apply_to_subsystem(a, v12, 1) ≈ kron(a, I2) * v12
                @test apply_to_subsystem(a, v12, 2) ≈ kron(I2, a) * v12
                @test apply_to_subsystem(a, v123, 1, [2, 2, 3]) ≈ kron(a, I6) * v123
                @test apply_to_subsystem(a, v123, 2, [2, 2, 3]) ≈ kron(I2, a, I3) * v123
                @test apply_to_subsystem(c, v123, 3, [2, 2, 3]) ≈ kron(I4, c) * v123
                @test apply_to_subsystem(ab, v12, [1, 2]) ≈ ab * v12
                @test apply_to_subsystem(ab, v12, [2, 1]) ≈ permute_systems(ab, [2, 1], [2, 2]) * v12
                @test apply_to_subsystem(ac, v123, [2, 3], [2, 2, 3]) ≈ kron(I2, ac) * v123
                @test apply_to_subsystem(bc, v123, [1, 3], [2, 2, 3]) ≈
                      permute_systems(kron(I2, bc), [2, 1, 3], [2, 2, 3]) * v123
                @test apply_to_subsystem(abc, v123, [2, 1, 3], [2, 2, 3]) ≈
                      permute_systems(abc, [2, 1, 3], [2, 2, 3]) * v123
            end
        end
        @testset "Rectangular op applied to ψ " begin
            model = JuMP.Model()
            H = [1 1; 1 -1]
            JuMP.@variable(model, ψ[1:4])
            res = Array{JuMP.AbstractJuMPScalar}(undef, 4)
            res[1:2] .= H * ψ[1:2]
            res[3:4] .= H * ψ[3:4]
            @test apply_to_subsystem(H, ψ, 2, [2, 2]) == res
            for R ∈ (Float64, BigFloat), T ∈ (R, Complex{R})
                a = randn(T, 3, 2)
                b = randn(T, 2, 3)
                c = randn(T, 4, 6)
                d = randn(T, 6, 4)
                e = randn(T, 1, 4)
                v2 = randn(T, 2)
                v3 = randn(T, 3)
                v4 = randn(T, 4)
                v6 = randn(T, 6)
                v12 = randn(T, 12)
                v16 = randn(T, 16)
                I2 = Matrix(one(T) * I, (2, 2))
                @test apply_to_subsystem(a, v2, 1, [2]) ≈ a * v2
                @test apply_to_subsystem(a, v4, 2, [2, 2]) ≈ kron(I2, a) * v4
                @test apply_to_subsystem(b, v6, 2, [2, 3]) ≈ kron(I2, b) * v6
                @test apply_to_subsystem(c, v12, [2, 3], [2, 2, 3]) ≈ kron(I2, c) * v12
                @test apply_to_subsystem(d, v16, [2, 3], [2, 2, 2, 2]) ≈ kron(I2, d, I2) * v16
                @test apply_to_subsystem(e, v16, [2, 3], [2, 2, 2, 2]) ≈ kron(I2, e, I2) * v16
            end
        end
        @testset "Square Kraus applied to ρ      " begin
            model = JuMP.Model()
            H = [1 1; 1 -1]
            JuMP.@variable(model, ρ[1:4, 1:4], Hermitian)
            res = Array{eltype(ρ)}(undef, 4, 4)
            res[1:2, 1:2] = H * ρ[1:2, 1:2] * H'
            res[1:2, 3:4] = H * ρ[1:2, 3:4] * H'
            res[3:4, 1:2] = H * ρ[3:4, 1:2] * H'
            res[3:4, 3:4] = H * ρ[3:4, 3:4] * H'
            @test apply_to_subsystem([H], ρ, 2, [2, 2]) == res
            d1, d2, d3 = 2, 2, 3
            for R ∈ (Float64, BigFloat), T ∈ (R, Complex{R})
                a = randn(T, d1, d1)
                b = randn(T, d2, d2)
                c = randn(T, d3, d3)
                ab = kron(a, b)
                ac = kron(a, c)
                bc = kron(b, c)
                abc = kron(ab, c)
                I2 = Matrix(one(T) * I, (2, 2))
                I3 = Matrix(one(T) * I, (3, 3))
                I4 = Matrix(one(T) * I, (4, 4))
                I6 = Matrix(one(T) * I, (6, 6))
                @test apply_to_subsystem([a], ab, 1) ≈ kron(a, I2) * ab * kron(a, I2)'
                @test apply_to_subsystem([a], ab, 2) ≈ kron(I2, a) * ab * kron(I2, a)'
                @test apply_to_subsystem([a], abc, 1, [2, 2, 3]) ≈ kron(a, I6) * abc * kron(a, I6)'
                @test apply_to_subsystem([a, b], abc, 2, [2, 2, 3]) ≈
                      kron(I2, a, I3) * abc * kron(I2, a, I3)' + kron(I2, b, I3) * abc * kron(I2, b, I3)'
                @test apply_to_subsystem([c], abc, 3, [2, 2, 3]) ≈ kron(I4, c) * abc * kron(I4, c)'
                @test apply_to_subsystem([ab], ab, [1, 2]) ≈ ab * ab * ab'
                @test apply_to_subsystem([ab], ab, [2, 1]) ≈
                      permute_systems(ab, [2, 1], [2, 2]) * ab * permute_systems(ab, [2, 1], [2, 2])'
                @test apply_to_subsystem([ac, bc], abc, [2, 3], [2, 2, 3]) ≈
                      kron(I2, ac) * abc * kron(I2, ac)' + kron(I2, bc) * abc * kron(I2, bc)'
                @test apply_to_subsystem([abc], abc, [2, 1, 3], [2, 2, 3]) ≈
                      permute_systems(abc, [2, 1, 3], [2, 2, 3]) * abc * permute_systems(abc, [2, 1, 3], [2, 2, 3])'

                #sparse arrays
                d = 3^4
                SparseM = SparseArrays.spdiagm(-1 => randn(T, d - 1), 1 => randn(T, d - 1))
                StdM = Matrix(SparseM)
                op1 = randn(T, 3^2, 3^2)
                op2 = randn(T, d, d)
                @test apply_to_subsystem([op1], SparseM, [3, 1], [3, 3, 3, 3]) ≈
                      apply_to_subsystem([op1], StdM, [3, 1], [3, 3, 3, 3])
                @test apply_to_subsystem([op2], SparseM, [3, 1, 4, 2], [3, 3, 3, 3]) ≈
                      apply_to_subsystem([op2], StdM, [3, 1, 4, 2], [3, 3, 3, 3])
            end
            for wrapper ∈ (Symmetric, Hermitian)
                a = randn(ComplexF64, d1, d1)
                b = randn(ComplexF64, d1 * d3, d1 * d3)
                M = wrapper(randn(ComplexF64, (d1 * d2 * d3, d1 * d2 * d3)))
                x = Matrix(M)
                @test apply_to_subsystem([a], M, 1, [2, 2, 3]) ≈ apply_to_subsystem([a], x, 1, [2, 2, 3])
                @test apply_to_subsystem([b], M, [1, 3], [2, 2, 3]) ≈ apply_to_subsystem([b], x, [1, 3], [2, 2, 3])
            end
        end
        @testset "Rectangular Kraus applied to ρ       " begin
            model = JuMP.Model()
            k = [1 1; 1 -1; 1 -1]
            JuMP.@variable(model, ρ[1:4, 1:4], Hermitian)
            res = Array{eltype(ρ)}(undef, 6, 6)
            res[1:3, 1:3] = k * ρ[1:2, 1:2] * k'
            res[1:3, 4:6] = k * ρ[1:2, 3:4] * k'
            res[4:6, 1:3] = k * ρ[3:4, 1:2] * k'
            res[4:6, 4:6] = k * ρ[3:4, 3:4] * k'
            @test apply_to_subsystem([k], ρ, 2, [2, 2]) == res
            d1, d2 = 2, 3
            for R ∈ (Float64, BigFloat), T ∈ (R, Complex{R})
                k1 = randn(T, d1, d2)
                k2 = randn(T, d1, d2)
                k3 = randn(T, d2, d1)
                k4 = randn(T, d2, d1)
                k5 = randn(T, d1 * d1, d2 * d2)
                k6 = randn(T, d2 * d2, d1 * d1)
                k7 = randn(T, 1, d1 * d1)

                a = randn(T, d1^3, d1^3)
                b = randn(T, d2^3, d2^3)
                c = randn(T, 3, 3)
                d = randn(T, 1, d1^3)

                I2 = Matrix(one(T) * I, (2, 2))
                I3 = Matrix(one(T) * I, (3, 3))
                I9 = Matrix(one(T) * I, (9, 9))
                @test apply_to_subsystem([k1, k2], c, 1, [3]) ≈ k1 * c * k1' + k2 * c * k2'
                @test apply_to_subsystem([k1, k2], b, 1, [3, 3, 3]) ≈
                      kron(k1, I9) * b * kron(k1, I9)' + kron(k2, I9) * b * kron(k2, I9)'
                @test apply_to_subsystem([k1, k2], b, 3, [3, 3, 3]) ≈
                      kron(I9, k1) * b * kron(I9, k1)' + kron(I9, k2) * b * kron(I9, k2)'
                @test apply_to_subsystem([k1, k2], b, 2, [3, 3, 3]) ≈
                      permute_systems(kron(k1, I9), [2, 1, 3], [2 3; 3 3; 3 3]) *
                      b *
                      permute_systems(kron(k1, I9), [2, 1, 3], [2 3; 3 3; 3 3])' +
                      permute_systems(kron(k2, I9), [2, 1, 3], [2 3; 3 3; 3 3]) *
                      b *
                      permute_systems(kron(k2, I9), [2, 1, 3], [2 3; 3 3; 3 3])'

                @test apply_to_subsystem([k5], b, [1, 2], [3, 3, 3]) ≈ kron(k5, I3) * b * kron(k5, I3)'
                @test apply_to_subsystem([k6], a, [2, 3], [2, 2, 2]) ≈ kron(I2, k6) * a * kron(I2, k6)'
                @test apply_to_subsystem([k7], a, [2, 3], [2, 2, 2]) ≈ kron(I2, k7) * a * kron(I2, k7)'

                #sparse arrays
                d = 3^4
                SparseM = SparseArrays.spdiagm(-1 => randn(T, d - 1), 1 => randn(T, d - 1))
                StdM = Matrix(SparseM)
                k1 = sparse(randn(T, 2^2, 3^2))
                k2 = sparse(randn(T, 2^2, 3^2))
                k3 = sparse(randn(T, 1, 3^2))
                k4 = sparse(randn(T, 1, 3^2))
                @test apply_to_subsystem([k1, k2], SparseM, [2, 3], [3, 3, 3, 3]) ≈
                      apply_to_subsystem([k1, k2], StdM, [2, 3], [3, 3, 3, 3])
                @test apply_to_subsystem([k3, k4], SparseM, [2, 3], [3, 3, 3, 3]) ≈
                      apply_to_subsystem([k3, k4], StdM, [2, 3], [3, 3, 3, 3])
            end
        end
    end
    @testset "Partial trace      " begin
        model = JuMP.Model()
        JuMP.@variable(model, ρ[1:4, 1:4], Symmetric)
        ptrace = [tr(ρ[1:2, 1:2]) tr(ρ[1:2, 3:4]); tr(ρ[3:4, 1:2]) tr(ρ[3:4, 3:4])]
        @test partial_trace(ρ, 2, [2, 2]) == ptrace
        JuMP.@variable(model, σ[1:4, 1:4], Hermitian)
        ptrace = [tr(σ[1:2, 1:2]) tr(σ[1:2, 3:4]); tr(σ[3:4, 1:2]) tr(σ[3:4, 3:4])]
        @test partial_trace(σ, 2, [2, 2]) == ptrace
        d1, d2, d3 = 2, 2, 3
        for R ∈ (Float64, BigFloat), T ∈ (R, Complex{R})
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
        for R ∈ (Float64, BigFloat), T ∈ (R, Complex{R})
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
            for R ∈ (Float64, BigFloat), T ∈ (R, Complex{R})
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

                @test permute_systems(uvw, [5, 4, 3, 2, 1], [d1, 1, d2, 1, d3]) ≈ kron(w, v, u)
                @test permute_systems(uvw, [3, 4, 5, 2, 1, 6], [d1, d2, d3, 1, 1, 1]) ≈ kron(w, v, u)
            end
        end

        @testset "Square matrices" begin
            d1, d2, d3 = 2, 2, 3
            for R ∈ (Float64, BigFloat), T ∈ (R, Complex{R})
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
            for R ∈ (Float64, BigFloat), T ∈ (R, Complex{R})
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
        a = ones(Int, 4, 4)
        a[1] = 2
        trrp = [
            3/2 0 1 0
            0 3/2 0 1
            1 0 1 0
            0 1 0 1
        ]
        @test trace_replace(a, 2) == trrp
        d1, d2, d3 = 2, 2, 3
        for R ∈ (Float64, BigFloat), T ∈ (R, Complex{R})
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
