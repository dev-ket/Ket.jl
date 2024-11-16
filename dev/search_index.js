var documenterSearchIndex = {"docs":
[{"location":"api/#List-of-functions","page":"List of functions","title":"List of functions","text":"","category":"section"},{"location":"api/#Basic","page":"List of functions","title":"Basic","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"ket\nketbra\nproj\nshift\nclock\npauli\ngell_mann\ngell_mann!\npartial_trace\npartial_transpose\npermute_systems!\npermute_systems\ncleanup!\nsymmetric_projection\northonormal_range\npermutation_matrix\nn_body_basis","category":"page"},{"location":"api/#Ket.ket","page":"List of functions","title":"Ket.ket","text":"ket([T=ComplexF64,] i::Integer, d::Integer = 2)\n\nProduces a ket of dimension d with nonzero element i.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.ketbra","page":"List of functions","title":"Ket.ketbra","text":"ketbra(v::AbstractVector)\n\nProduces a ketbra of vector v.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.proj","page":"List of functions","title":"Ket.proj","text":"proj([T=ComplexF64,] i::Integer, d::Integer = 2)\n\nProduces a projector onto the basis state i in dimension d.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.shift","page":"List of functions","title":"Ket.shift","text":"shift([T=ComplexF64,] d::Integer, p::Integer = 1)\n\nConstructs the shift operator X of dimension d to the power p.\n\nReference: Generalized Clifford algebra\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.clock","page":"List of functions","title":"Ket.clock","text":"clock([T=ComplexF64,] d::Integer, p::Integer = 1)\n\nConstructs the clock operator Z of dimension d to the power p.\n\nReference: Generalized Clifford algebra\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.pauli","page":"List of functions","title":"Ket.pauli","text":"pauli([T=ComplexF64,], ind::Vector{<:Integer})\n\nConstructs the Pauli matrices: 0 or \"I\" for the identity, 1 or \"X\" for the Pauli X operation, 2 or \"Y\" for the Pauli Y operator, and 3 or \"Z\" for the Pauli Z operator. Vectors of integers between 0 and 3 or strings of I, X, Y, Z automatically generate Kronecker products of the corresponding operators.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.gell_mann","page":"List of functions","title":"Ket.gell_mann","text":"gell_mann([T=ComplexF64,], d::Integer = 3)\n\nConstructs the set G of generalized Gell-Mann matrices in dimension d such that G[1] = I and Tr(G[i]*G[j]) = 2 δ_ij.\n\nReference: Generalizations of Pauli matrices\n\n\n\n\n\ngell_mann([T=ComplexF64,], i::Integer, j::Integer, d::Integer = 3)\n\nConstructs the set i,jth Gell-Mann matrix of dimension d.\n\nReference: Generalizations of Pauli matrices\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.gell_mann!","page":"List of functions","title":"Ket.gell_mann!","text":"gell_mann!(res::AbstractMatrix{T}, i::Integer, j::Integer, d::Integer = 3)\n\nIn-place version of gell_mann.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.partial_trace","page":"List of functions","title":"Ket.partial_trace","text":"partial_trace(X::AbstractMatrix, remove::AbstractVector, dims::AbstractVector = _equal_sizes(X))\n\nTakes the partial trace of matrix X with subsystem dimensions dims over the subsystems in remove. If the argument dims is omitted two equally-sized subsystems are assumed.\n\n\n\n\n\npartial_trace(X::AbstractMatrix, remove::Integer, dims::AbstractVector = _equal_sizes(X)))\n\nTakes the partial trace of matrix X with subsystem dimensions dims over the subsystem remove. If the argument dims is omitted two equally-sized subsystems are assumed.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.partial_transpose","page":"List of functions","title":"Ket.partial_transpose","text":"partial_transpose(X::AbstractMatrix, transp::AbstractVector, dims::AbstractVector = _equal_sizes(X))\n\nTakes the partial transpose of matrix X with subsystem dimensions dims on the subsystems in transp. If the argument dims is omitted two equally-sized subsystems are assumed.\n\n\n\n\n\npartial_transpose(X::AbstractMatrix, transp::Integer, dims::AbstractVector = _equal_sizes(X))\n\nTakes the partial transpose of matrix X with subsystem dimensions dims on the subsystem transp. If the argument dims is omitted two equally-sized subsystems are assumed.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.permute_systems!","page":"List of functions","title":"Ket.permute_systems!","text":"permute_systems!(X::AbstractVector, perm::AbstractVector, dims::AbstractVector = _equal_sizes(X))\n\nPermutes the order of the subsystems of vector X with subsystem dimensions dims in-place according to the permutation perm. If the argument dims is omitted two equally-sized subsystems are assumed.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.permute_systems","page":"List of functions","title":"Ket.permute_systems","text":"permute_systems(X::AbstractMatrix, perm::AbstractVector, dims::AbstractVector = _equal_sizes(X))\n\nPermutes the order of the subsystems of the square matrix X, which is composed by square subsystems of dimensions dims, according to the permutation perm. If the argument dims is omitted two equally-sized subsystems are assumed.\n\n\n\n\n\npermute_systems(X::AbstractMatrix, perm::Vector, dims::Matrix)\n\nPermutes the order of the subsystems of the matrix X, which is composed by subsystems of dimensions dims, according to the permutation perm. dims should be a n x 2 matrix where dims[i, 1] is the number of rows of subsystem i, and dims[i,2] is its number of columns. \n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.cleanup!","page":"List of functions","title":"Ket.cleanup!","text":"cleanup!(M::AbstractArray{T}; tol = Base.rtoldefault(real(T)))\n\nZeroes out real or imaginary parts of M that are smaller than tol.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.symmetric_projection","page":"List of functions","title":"Ket.symmetric_projection","text":"symmetric_projection(dim::Integer, n::Integer; partial::Bool=true)\n\nOrthogonal projection onto the symmetric subspace of n copies of a dim-dimensional space. By default (partial=true) it returns an isometry (say, V) encoding the symmetric subspace. If partial=false, then it returns the actual projection V * V'.\n\nReference: Watrous' book, Sec. 7.1.1\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.orthonormal_range","page":"List of functions","title":"Ket.orthonormal_range","text":"orthonormal_range(A::AbstractMatrix{T}; mode::Integer=nothing, tol::T=nothing, sp::Bool=true) where {T<:Number}\n\nOrthonormal basis for the range of A. When A is sparse (or mode = 0), uses a QR factorization and returns a sparse result, otherwise uses an SVD and returns a dense matrix (mode = 1). Input A will be overwritten during the factorization. Tolerance tol is used to compute the rank and is automatically set if not provided.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.permutation_matrix","page":"List of functions","title":"Ket.permutation_matrix","text":"permutation_matrix(dims::Union{Integer,AbstractVector}, perm::AbstractVector)\n\nUnitary that permutes subsystems of dimension dims according to the permutation perm. If dims is an Integer, assumes there are length(perm) subsystems of equal dimensions dims.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.n_body_basis","page":"List of functions","title":"Ket.n_body_basis","text":"n_body_basis(\nn::Integer,\nn_parties::Integer;\nsb::AbstractVector{<:AbstractMatrix} = [pauli(1), pauli(2), pauli(3)],\nsparse::Bool = true,\neye::AbstractMatrix = I(size(sb[1], 1))\n\nReturn the basis of n nontrivial operators acting on n_parties, by default using Pauli matrices.\n\nFor example, n_body_basis(2, 3) generate all products of two Paulis and one identity, so  X  X  1 X  1  X  X  Y  1  1  Z  Z.\n\nInstead of Paulis, a basis can be provided by the parameter sb, and the identity can be changed with eye. If sparse is true, the resulting basis will use sparse matrices, otherwise it will agree with sb.\n\nThis function returns a generator, which can then be used e.g. in for loops without fully allocating the entire basis at once. If you need a vector, call collect on it.\n\n\n\n\n\n","category":"function"},{"location":"api/#Entropy","page":"List of functions","title":"Entropy","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"entropy\nbinary_entropy\nrelative_entropy\nbinary_relative_entropy\nconditional_entropy","category":"page"},{"location":"api/#Ket.entropy","page":"List of functions","title":"Ket.entropy","text":"entropy([base=2,] ρ::AbstractMatrix)\n\nComputes the von Neumann entropy -tr(ρ log ρ) of a positive semidefinite operator ρ using a base base logarithm.\n\nReference: von Neumann entropy.\n\n\n\n\n\nentropy([base=2,] p::AbstractVector)\n\nComputes the Shannon entropy -Σᵢpᵢlog(pᵢ) of a non-negative vector p using a base base logarithm.\n\nReference: Entropy (information theory).\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.binary_entropy","page":"List of functions","title":"Ket.binary_entropy","text":"binary_entropy([base=2,] p::Real)\n\nComputes the Shannon entropy -p log(p) - (1-p)log(1-p) of a probability p using a base base logarithm.\n\nReference: Entropy (information theory).\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.relative_entropy","page":"List of functions","title":"Ket.relative_entropy","text":"relative_entropy([base=2,] ρ::AbstractMatrix, σ::AbstractMatrix)\n\nComputes the (quantum) relative entropy tr(ρ (log ρ - log σ)) between positive semidefinite matrices ρ and σ using a base base logarithm. Note that the support of ρ must be contained in the support of σ but for efficiency this is not checked.\n\nReference: Quantum relative entropy.\n\n\n\n\n\nrelative_entropy([base=2,] p::AbstractVector, q::AbstractVector)\n\nComputes the relative entropy D(p||q) = Σᵢpᵢlog(pᵢ/qᵢ) between two non-negative vectors p and q using a base base logarithm. Note that the support of p must be contained in the support of q but for efficiency this is not checked.\n\nReference: Relative entropy.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.binary_relative_entropy","page":"List of functions","title":"Ket.binary_relative_entropy","text":"binary_relative_entropy([base=2,] p::Real, q::Real)\n\nComputes the binary relative entropy D(p||q) = p log(p/q) + (1-p) log((1-p)/(1-q)) between two probabilities p and q using a base base logarithm.\n\nReference: Relative entropy.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.conditional_entropy","page":"List of functions","title":"Ket.conditional_entropy","text":"conditional_entropy([base=2,] pAB::AbstractMatrix)\n\nComputes the conditional Shannon entropy H(A|B) of the joint probability distribution pAB using a base base logarithm.\n\nReference: Conditional entropy.\n\n\n\n\n\nconditional_entropy([base=2,], rho::AbstractMatrix, csys::AbstractVector, dims::AbstractVector)\n\nComputes the conditional von Neumann entropy of rho with subsystem dimensions dims and conditioning systems csys, using a base base logarithm.\n\nReference: Conditional quantum entropy.\n\n\n\n\n\n","category":"function"},{"location":"api/#Entanglement","page":"List of functions","title":"Entanglement","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"schmidt_decomposition\nentanglement_entropy\nrandom_robustness\nschmidt_number\nppt_mixture","category":"page"},{"location":"api/#Ket.schmidt_decomposition","page":"List of functions","title":"Ket.schmidt_decomposition","text":"schmidt_decomposition(ψ::AbstractVector, dims::AbstractVector{<:Integer} = _equal_sizes(ψ))\n\nProduces the Schmidt decomposition of ψ with subsystem dimensions dims. If the argument dims is omitted equally-sized subsystems are assumed. Returns the (sorted) Schmidt coefficients λ and isometries U, V such that kron(U', V')*ψ is of Schmidt form.\n\nReference: Schmidt decomposition.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.entanglement_entropy","page":"List of functions","title":"Ket.entanglement_entropy","text":"entanglement_entropy(ψ::AbstractVector, dims::AbstractVector{<:Integer} = _equal_sizes(ψ))\n\nComputes the relative entropy of entanglement of a bipartite pure state ψ with subsystem dimensions dims. If the argument dims is omitted equally-sized subsystems are assumed.\n\n\n\n\n\nentanglement_entropy(ρ::AbstractMatrix, dims::AbstractVector = _equal_sizes(ρ), n::Integer = 1)\n\nLower bounds the relative entropy of entanglement of a bipartite state ρ with subsystem dimensions dims using level n of the DPS hierarchy. If the argument dims is omitted equally-sized subsystems are assumed.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.random_robustness","page":"List of functions","title":"Ket.random_robustness","text":"random_robustness(\nρ::AbstractMatrix{T},\ndims::AbstractVector{<:Integer} = _equal_sizes(ρ),\nn::Integer = 1;\nppt::Bool = true,\nverbose::Bool = false,\nsolver = Hypatia.Optimizer{_solver_type(T)})\n\nLower bounds the random robustness of state ρ with subsystem dimensions dims using level n of the DPS hierarchy. Argument ppt indicates whether to include the partial transposition constraints.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.schmidt_number","page":"List of functions","title":"Ket.schmidt_number","text":"schmidt_number(\n    ρ::AbstractMatrix{T},\n    s::Integer = 2,\n    dims::AbstractVector{<:Integer} = _equal_sizes(ρ),\n    n::Integer = 1;\n    ppt::Bool = true,\n    verbose::Bool = false,\n    solver = Hypatia.Optimizer{_solver_type(T)})\n\nUpper bound on the random robustness of ρ such that it has a Schmidt number s.\n\nIf a state ρ with local dimensions d_A and d_B has Schmidt number s, then there is a PSD matrix ω in the extended space AABB, where A and B^ have dimension s, such that ω  s is separable  against AABB and Π ω Π = ρ, where Π = 1_A  s ψ^+  1_B, and ψ^+ is a non-normalized maximally entangled state. Separabiity is tested with the DPS hierarchy, with n controlling the how many copies of the BB subsystem are used. \n\nReferences:     Hulpke, Bruss, Lewenstein, Sanpera arXiv:quant-ph/0401118Weilenmann, Dive, Trillo, Aguilar, Navascués arXiv:1912.10056\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.ppt_mixture","page":"List of functions","title":"Ket.ppt_mixture","text":"function ppt_mixture(\nρ::AbstractMatrix{T},\ndims::AbstractVector{<:Integer};\nverbose::Bool = false,\nsolver = Hypatia.Optimizer{_solver_type(T)})\n\nLower bound on the white noise such that ρ is still a genuinely multipartite entangled state and a GME witness that detects ρ.\n\nThe set of GME states is approximated by the set of PPT mixtures, so the entanglement across the bipartitions is decided with the PPT criterion. If the state is a PPT mixture, returns a 0 matrix instead of a witness.\n\nReference: Jungnitsch, Moroder, Guehne arXiv:quant-ph/0401118\n\n\n\n\n\nfunction ppt_mixture(\nρ::AbstractMatrix{T},\ndims::AbstractVector{<:Integer},\nobs::AbstractVector{<:AbstractMatrix} = Vector{Matrix}();\nverbose::Bool = false,\nsolver = Hypatia.Optimizer{_solver_type(T)})\n\nLower bound on the white noise such that ρ is still a genuinely multipartite entangled state that can be detected with a witness using only the operators provided in obs, and the values of the coefficients defining such a witness.\n\nMore precisely, if a list of observables O_i is provided in the parameter obs, the witness will be of the form _i α_i O_i and detects ρ only using these observables. For example, using only two-body operators (and lower order) one can call \n\njulia> two_body_basis = collect(Iterators.flatten(n_body_basis(i, 3) for i in 0:2))\njulia> ppt_mixture(state_ghz(2, 3), [2, 2, 2], two_body_basis)\n\nReference: Jungnitsch, Moroder, Guehne arXiv:quant-ph/0401118\n\n\n\n\n\n","category":"function"},{"location":"api/#Measurements","page":"List of functions","title":"Measurements","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"sic_povm\ntest_sic\ntest_povm\ndilate_povm\npovm\nmub\ntest_mub","category":"page"},{"location":"api/#Ket.sic_povm","page":"List of functions","title":"Ket.sic_povm","text":"sic_povm([T=ComplexF64,] d::Integer)\n\nConstructs a vector of d² vectors |vᵢ⟩ such that |vᵢ⟩⟨vᵢ| forms a SIC-POVM of dimension d. This construction is based on the Weyl-Heisenberg fiducial.\n\nReference: Appleby, Yadsan-Appleby, Zauner, arXiv:1209.1813\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.test_sic","page":"List of functions","title":"Ket.test_sic","text":"test_sic(vecs)\n\nChecks if vecs is a vector of d² vectors |vᵢ⟩ such that |vᵢ⟩⟨vᵢ| forms a SIC-POVM of dimension d.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.test_povm","page":"List of functions","title":"Ket.test_povm","text":"test_povm(A::Vector{<:AbstractMatrix{T}})\n\nChecks if the measurement defined by A is valid (hermitian, semi-definite positive, and normalized).\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.dilate_povm","page":"List of functions","title":"Ket.dilate_povm","text":"dilate_povm(vecs::Vector{Vector{T}})\n\nDoes the Naimark dilation of a rank-1 POVM given as a vector of vectors. This is the minimal dilation.\n\n\n\n\n\ndilate_povm(E::Vector{<:AbstractMatrix})\n\nDoes the Naimark dilation of a POVM given as a vector of matrices. This always works, but is wasteful if the POVM elements are not full rank.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.povm","page":"List of functions","title":"Ket.povm","text":"povm(B::Vector{<:AbstractMatrix{T}})\n\nCreates a set of (projective) measurements from a set of bases given as unitary matrices.\n\n\n\n\n\npovm(A::Array{T, 4}, n::Vector{Int64})\n\nConverts a set of measurements in the common tensor format into a matrix of matrices. The second argument is fixed by the size of A but can also contain custom number of outcomes.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.mub","page":"List of functions","title":"Ket.mub","text":"mub([T=ComplexF64,] d::Integer)\n\nConstruction of the standard complete set of MUBs. The output contains 1+minᵢ pᵢ^rᵢ bases, where d = ∏ᵢ pᵢ^rᵢ.\n\nReference: Durt, Englert, Bengtsson, Życzkowski, arXiv:1004.3348.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.test_mub","page":"List of functions","title":"Ket.test_mub","text":"test_mub(B::Vector{Matrix{<:Number}})\n\nChecks if the input bases are mutually unbiased.\n\n\n\n\n\n","category":"function"},{"location":"api/#Incompatibility","page":"List of functions","title":"Incompatibility","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"incompatibility_robustness_depolarizing","category":"page"},{"location":"api/#Ket.incompatibility_robustness_depolarizing","page":"List of functions","title":"Ket.incompatibility_robustness_depolarizing","text":"incompatibility_robustness_depolarizing(A::Vector{Measurement{<:Number}})\n\nComputes the incompatibility depolarizing robustness of the measurements in the vector A.\n\nReference: Designolle, Farkas, Kaniewski, arXiv:1906.00448\n\n\n\n\n\n","category":"function"},{"location":"api/#Nonlocality","page":"List of functions","title":"Nonlocality","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"chsh\ncglmp\ninn22\nlocal_bound\ntsirelson_bound\nseesaw\ncorrelation_tensor\nprobability_tensor\nfp2cg","category":"page"},{"location":"api/#Ket.chsh","page":"List of functions","title":"Ket.chsh","text":"chsh([T=Float64,] d::Integer = 2)\n\nCHSH-d nonlocal game in full probability notation. If T is an integer type the game is unnormalized.\n\nReference: Buhrman and Massar, arXiv:quant-ph/0409066.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.cglmp","page":"List of functions","title":"Ket.cglmp","text":"cglmp([T=Float64,] d::Integer)\n\nCGLMP nonlocal game in full probability notation. If T is an integer type the game is unnormalized.\n\nReferences: arXiv:quant-ph/0106024 for the original game, and arXiv:2005.13418 for the form presented here.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.inn22","page":"List of functions","title":"Ket.inn22","text":"inn22([T=Float64,] n::Integer = 3)\n\ninn22 Bell functional in Collins-Gisin notation. Local bound 1.\n\nReference: Śliwa, arXiv:quant-ph/0305190\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.local_bound","page":"List of functions","title":"Ket.local_bound","text":"local_bound(G::Array{T,4})\n\nComputes the local bound of a bipartite Bell functional G, written in full probability notation as a 4-dimensional array.\n\nReference: Araújo, Hirsch, and Quintino, arXiv:2005.13418.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.tsirelson_bound","page":"List of functions","title":"Ket.tsirelson_bound","text":"tsirelson_bound(CG::Matrix, scenario::Vector, level::Integer)\n\nUpper bounds the Tsirelson bound of a bipartite Bell funcional game CG, written in Collins-Gisin notation. scenario is a vector detailing the number of inputs and outputs, in the order [oa, ob, ia, ib]. level is an integer determining the level of the NPA hierarchy.\n\nThis function requires Moment. It is only available if you first do \"import MATLAB\" or \"using MATLAB\".\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.seesaw","page":"List of functions","title":"Ket.seesaw","text":"seesaw(CG::Matrix, scenario::Vector, d::Integer)\n\nMaximizes bipartite Bell functional in Collins-Gisin notation CG using the seesaw heuristic. scenario is a vector detailing the number of inputs and outputs, in the order [oa, ob, ia, ib]. d is an integer determining the local dimension of the strategy.\n\nIf oa == ob == 2 the heuristic reduces to a bunch of eigenvalue problems. Otherwise semidefinite programming is needed and we use the assemblage version of seesaw.\n\nReferences: Pál and Vértesi, arXiv:1006.3032,  section II.B.1 of Tavakoli et al., arXiv:2307.02551\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.correlation_tensor","page":"List of functions","title":"Ket.correlation_tensor","text":"correlation_tensor(p::AbstractArray{T, N2}; marg::Bool = true)\n\nApplies N sets of measurements onto a state rho to form a probability array. Convert a 2x...x2xmx...xm probability array into\n\na mx...xm correlation array (no marginals)\na (m+1)x...x(m+1) correlation array (marginals).\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.probability_tensor","page":"List of functions","title":"Ket.probability_tensor","text":"probability_tensor(rho::Hermitian, all_Aax::Vector{Measurement}...)\n\nApplies N sets of measurements onto a state rho to form a probability array.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.fp2cg","page":"List of functions","title":"Ket.fp2cg","text":"fp2cg(V::Array{T,4}) where {T <: Real}\n\nTakes a bipartite Bell functional V in full probability notation and transforms it to Collins-Gisin notation.\n\n\n\n\n\n","category":"function"},{"location":"api/#Norms","page":"List of functions","title":"Norms","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"trace_norm\nkyfan_norm\nschatten_norm\ndiamond_norm","category":"page"},{"location":"api/#Ket.trace_norm","page":"List of functions","title":"Ket.trace_norm","text":"trace_norm(X::AbstractMatrix)\n\nComputes trace norm of matrix X.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.kyfan_norm","page":"List of functions","title":"Ket.kyfan_norm","text":"kyfan_norm(X::AbstractMatrix, k::Integer, p::Real = 2)\n\nComputes Ky-Fan (k,p) norm of matrix X.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.schatten_norm","page":"List of functions","title":"Ket.schatten_norm","text":"schatten_norm(X::AbstractMatrix, p::Real)\n\nComputes Schatten p-norm of matrix X.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.diamond_norm","page":"List of functions","title":"Ket.diamond_norm","text":"diamond_norm(J::AbstractMatrix, dims::AbstractVector)\n\nComputes the diamond norm of the supermap J given in the Choi-Jamiołkowski representation, with subsystem dimensions dims.\n\nReference: Diamond norm\n\n\n\n\n\ndiamond_norm(K::Vector{<:AbstractMatrix})\n\nComputes the diamond norm of the CP map given by the Kraus operators K.\n\n\n\n\n\n","category":"function"},{"location":"api/#Random","page":"List of functions","title":"Random","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"random_state\nrandom_state_ket\nrandom_unitary\nrandom_povm\nrandom_probability","category":"page"},{"location":"api/#Ket.random_state","page":"List of functions","title":"Ket.random_state","text":"random_state([T=ComplexF64,] d::Integer, k::Integer = d)\n\nProduces a uniformly distributed random quantum state in dimension d with rank k.\n\nReference: Życzkowski and Sommers, arXiv:quant-ph/0012101.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.random_state_ket","page":"List of functions","title":"Ket.random_state_ket","text":"random_state_ket([T=ComplexF64,] d::Integer)\n\nProduces a Haar-random quantum state vector in dimension d.\n\nReference: Życzkowski and Sommers, arXiv:quant-ph/0012101.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.random_unitary","page":"List of functions","title":"Ket.random_unitary","text":"random_unitary([T=ComplexF64,] d::Integer)\n\nProduces a Haar-random unitary matrix in dimension d. If T is a real type the output is instead a Haar-random (real) orthogonal matrix.\n\nReference: Stewart, doi:10.1137/0717034.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.random_povm","page":"List of functions","title":"Ket.random_povm","text":"random_povm([T=ComplexF64,] d::Integer, n::Integer, r::Integer)\n\nProduces a random POVM of dimension d with n outcomes and rank min(k, d).\n\nReference: Heinosaari et al., arXiv:1902.04751.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.random_probability","page":"List of functions","title":"Ket.random_probability","text":"random_probability([T=Float64,] d::Integer)\n\nProduces a random probability vector of dimension d uniformly distributed on the simplex.\n\nReference: Dirichlet distribution\n\n\n\n\n\n","category":"function"},{"location":"api/#States","page":"List of functions","title":"States","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"state_phiplus_ket\nstate_phiplus\nisotropic\nstate_psiminus_ket\nstate_psiminus\nstate_super_singlet_ket\nstate_super_singlet\nstate_ghz_ket\nstate_ghz\nstate_w_ket\nstate_w\nwhite_noise\nwhite_noise!","category":"page"},{"location":"api/#Ket.state_phiplus_ket","page":"List of functions","title":"Ket.state_phiplus_ket","text":"state_phiplus_ket([T=ComplexF64,] d::Integer = 2)\n\nProduces the vector of the maximally entangled state Φ⁺ of local dimension d.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.state_phiplus","page":"List of functions","title":"Ket.state_phiplus","text":"state_phiplus([T=ComplexF64,] d::Integer = 2; v::Real = 1)\n\nProduces the maximally entangled state Φ⁺ of local dimension d with visibility v.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.isotropic","page":"List of functions","title":"Ket.isotropic","text":"isotropic(v::Real, d::Integer = 2)\n\nProduces the isotropic state of local dimension d with visibility v.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.state_psiminus_ket","page":"List of functions","title":"Ket.state_psiminus_ket","text":"state_psiminus_ket([T=ComplexF64,] d::Integer = 2)\n\nProduces the vector of the maximally entangled state ψ⁻ of local dimension d.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.state_psiminus","page":"List of functions","title":"Ket.state_psiminus","text":"state_psiminus([T=ComplexF64,] d::Integer = 2; v::Real = 1)\n\nProduces the maximally entangled state ψ⁻ of local dimension d with visibility v.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.state_super_singlet_ket","page":"List of functions","title":"Ket.state_super_singlet_ket","text":"state_super_singlet_ket([T=ComplexF64,] N::Integer = 3; coeff = 1/√N!)\n\nProduces the vector of the N-partite N-level singlet state.\n\nReference: Adán Cabello, arXiv:quant-ph/0203119\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.state_super_singlet","page":"List of functions","title":"Ket.state_super_singlet","text":"state_super_singlet([T=ComplexF64,] N::Integer = 3; v::Real = 1, coeff = 1/√d)\n\nProduces the N-partite N-level singlet state with visibility v.\n\nReference: Adán Cabello, arXiv:quant-ph/0203119\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.state_ghz_ket","page":"List of functions","title":"Ket.state_ghz_ket","text":"state_ghz_ket([T=ComplexF64,] d::Integer = 2, N::Integer = 3; coeff = 1/√d)\n\nProduces the vector of the GHZ state local dimension d.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.state_ghz","page":"List of functions","title":"Ket.state_ghz","text":"state_ghz([T=ComplexF64,] d::Integer = 2, N::Integer = 3; v::Real = 1, coeff = 1/√d)\n\nProduces the GHZ state of local dimension d with visibility v.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.state_w_ket","page":"List of functions","title":"Ket.state_w_ket","text":"state_w_ket([T=ComplexF64,] N::Integer = 3; coeff = 1/√d)\n\nProduces the vector of the N-partite W state.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.state_w","page":"List of functions","title":"Ket.state_w","text":"state_w([T=ComplexF64,] N::Integer = 3; v::Real = 1, coeff = 1/√d)\n\nProduces the N-partite W state with visibility v.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.white_noise","page":"List of functions","title":"Ket.white_noise","text":"white_noise(rho::AbstractMatrix, v::Real)\n\nReturns v * rho + (1 - v) * id, where id is the maximally mixed state.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.white_noise!","page":"List of functions","title":"Ket.white_noise!","text":"white_noise!(rho::AbstractMatrix, v::Real)\n\nModifies rho in place to tranform in into v * rho + (1 - v) * id where id is the maximally mixed state.\n\n\n\n\n\n","category":"function"},{"location":"api/#Supermaps","page":"List of functions","title":"Supermaps","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"choi","category":"page"},{"location":"api/#Ket.choi","page":"List of functions","title":"Ket.choi","text":"choi(K::Vector{<:AbstractMatrix})\n\nConstructs the Choi-Jamiołkowski representation of the CP map given by the Kraus operators K. The convention used is that choi(K) = ∑ᵢⱼ |i⟩⟨j|⊗K|i⟩⟨j|K'\n\n\n\n\n\n","category":"function"},{"location":"api/#Internal-functions","page":"List of functions","title":"Internal functions","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"Ket._partition\nKet._fiducial_WH\nKet._idx\nKet._tidx\nKet._idxperm","category":"page"},{"location":"api/#Ket._partition","page":"List of functions","title":"Ket._partition","text":"partition(n::Integer, k::Integer)\n\nIf n ≥ k partitions the set 1:n into k parts as equally sized as possible. Otherwise partitions it into n parts of size 1.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket._fiducial_WH","page":"List of functions","title":"Ket._fiducial_WH","text":"_fiducial_WH([T=ComplexF64,] d::Integer)\n\nComputes the fiducial Weyl-Heisenberg vector of dimension d.\n\nReference: Appleby, Yadsan-Appleby, Zauner, arXiv:1209.1813 http://www.gerhardzauner.at/sicfiducials.html\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket._idx","page":"List of functions","title":"Ket._idx","text":"_idx(tidx::Vector, dims::Vector)\n\nConverts a tensor index tidx = [i₁, i₂, ...] with subsystems dimensions dims to a standard index.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket._tidx","page":"List of functions","title":"Ket._tidx","text":"_tidx(idx::Integer, dims::Vector)\n\nConverts a standard index idx to a tensor index [i₁, i₂, ...] with subsystems dimensions dims.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket._idxperm","page":"List of functions","title":"Ket._idxperm","text":"_idxperm(perm::Vector, dims::Vector)\n\nComputes the index permutation associated with permuting the subsystems of a vector with subsystem dimensions dims according to perm.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"EditURL = \"https://github.com/araujoms/Ket.jl/blob/master/README.md\"","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: Banner)","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: Dev)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Toolbox for quantum information, nonlocality, and entanglement.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Highlights are the functions mub and sic_povm, that produce respectively MUBs and SIC-POVMs with arbitrary precision, local_bound that uses a parallelized algorithm to compute the local bound of a Bell inequality, and partial_trace and partial_transpose, that compute the partial trace and partial transpose in a way that can be used for optimization with JuMP. Also worth mentioning are the functions to produce uniformly-distributed random states, unitaries, and POVMs: random_state, random_unitary, random_povm. And the eponymous ket, of course.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For the full list of functions see the documentation.","category":"page"}]
}
