var documenterSearchIndex = {"docs":
[{"location":"api/#List-of-functions","page":"List of functions","title":"List of functions","text":"","category":"section"},{"location":"api/#Basic","page":"List of functions","title":"Basic","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"ket\nketbra\nproj\nshift\nclock\ngell_mann\ngell_mann!\npartial_trace\npartial_transpose\ncleanup!","category":"page"},{"location":"api/#Ket.ket","page":"List of functions","title":"Ket.ket","text":"ket([T=ComplexF64,] i::Integer, d::Integer = 2)\n\nProduces a ket of dimension d with nonzero element i.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.ketbra","page":"List of functions","title":"Ket.ketbra","text":"ketbra(v::AbstractVector)\n\nProduces a ketbra of vector v.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.proj","page":"List of functions","title":"Ket.proj","text":"proj([T=ComplexF64,] i::Integer, d::Integer = 2)\n\nProduces a projector onto the basis state i in dimension d.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.shift","page":"List of functions","title":"Ket.shift","text":"shift([T=ComplexF64,] d::Integer, p::Integer = 1)\n\nConstructs the shift operator X of dimension d to the power p.\n\nReference: Generalized Clifford algebra\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.clock","page":"List of functions","title":"Ket.clock","text":"clock([T=ComplexF64,] d::Integer, p::Integer = 1)\n\nConstructs the clock operator Z of dimension d to the power p.\n\nReference: Generalized Clifford algebra\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.gell_mann","page":"List of functions","title":"Ket.gell_mann","text":"gell_mann([T=ComplexF64,], d::Integer = 3)\n\nConstructs the set G of generalized Gell-Mann matrices in dimension d such that G[1] = I and G[i]*G[j] = 2 δ_ij.\n\nReference: Generalizations of Pauli matrices\n\n\n\n\n\ngell_mann([T=ComplexF64,], i::Integer, j::Integer, d::Integer = 3)\n\nConstructs the set i,jth Gell-Mann matrix of dimension d.\n\nReference: Generalizations of Pauli matrices\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.gell_mann!","page":"List of functions","title":"Ket.gell_mann!","text":"gell_mann!(res::AbstractMatrix{T}, i::Integer, j::Integer, d::Integer = 3)\n\nIn-place version of gell_mann.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.partial_trace","page":"List of functions","title":"Ket.partial_trace","text":"partial_trace(X::AbstractMatrix, remove::Vector, dims::Vector)\n\nTakes the partial trace of matrix X with subsystem dimensions dims over the subsystems in remove.\n\n\n\n\n\npartial_trace(X::AbstractMatrix, remove::Integer, dims::Vector)\n\nTakes the partial trace of matrix X with subsystem dimensions dims over the subsystem remove.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.partial_transpose","page":"List of functions","title":"Ket.partial_transpose","text":"partial_transpose(X::AbstractMatrix, transp::Vector, dims::Vector)\n\nTakes the partial transpose of matrix X with subsystem dimensions dims on the subsystems in transp.\n\n\n\n\n\npartial_transpose(X::AbstractMatrix, transp::Integer, dims::Vector)\n\nTakes the partial transpose of matrix X with subsystem dimensions dims on the subsystem transp.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.cleanup!","page":"List of functions","title":"Ket.cleanup!","text":"cleanup!(M::AbstractArray{T}; tol = Base.rtoldefault(real(T)))\n\nZeroes out real or imaginary parts of M that are smaller than tol.\n\n\n\n\n\n","category":"function"},{"location":"api/#Entropy","page":"List of functions","title":"Entropy","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"entropy\nbinary_entropy\nrelative_entropy\nbinary_relative_entropy\nconditional_entropy","category":"page"},{"location":"api/#Ket.entropy","page":"List of functions","title":"Ket.entropy","text":"entropy([base=2,] ρ::AbstractMatrix)\n\nComputes the von Neumann entropy -tr(ρ log ρ) of a positive semidefinite operator ρ using a base base logarithm.\n\nReference: von Neumann entropy.\n\n\n\n\n\nentropy([base=2,] p::AbstractVector)\n\nComputes the Shannon entropy -Σᵢpᵢlog(pᵢ) of a non-negative vector p using a base base logarithm.\n\nReference: Entropy (information theory).\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.binary_entropy","page":"List of functions","title":"Ket.binary_entropy","text":"binary_entropy([base=2,] p::Real)\n\nComputes the Shannon entropy -p log(p) - (1-p)log(1-p) of a probability p using a base base logarithm.\n\nReference: Entropy (information theory).\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.relative_entropy","page":"List of functions","title":"Ket.relative_entropy","text":"relative_entropy([base=2,] ρ::AbstractMatrix, σ::AbstractMatrix)\n\nComputes the (quantum) relative entropy tr(ρ (log ρ - log σ)) between positive semidefinite matrices ρ and σ using a base base logarithm. Note that the support of ρ must be contained in the support of σ but for efficiency this is not checked.\n\nReference: Quantum relative entropy.\n\n\n\n\n\nrelative_entropy([base=2,] p::AbstractVector, q::AbstractVector)\n\nComputes the relative entropy D(p||q) = Σᵢpᵢlog(pᵢ/qᵢ) between two non-negative vectors p and q using a base base logarithm. Note that the support of p must be contained in the support of q but for efficiency this is not checked.\n\nReference: Relative entropy.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.binary_relative_entropy","page":"List of functions","title":"Ket.binary_relative_entropy","text":"binary_relative_entropy([base=2,] p::Real, q::Real)\n\nComputes the binary relative entropy D(p||q) = p log(p/q) + (1-p) log((1-p)/(1-q)) between two probabilities p and q using a base base logarithm.\n\nReference: Relative entropy.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.conditional_entropy","page":"List of functions","title":"Ket.conditional_entropy","text":"conditional_entropy([base=2,] pAB::AbstractMatrix)\n\nComputes the conditional (Shannon) entropy H(A|B) of the joint probability distribution pAB using a base base logarithm.\n\nReference: Conditional entropy.\n\n\n\n\n\n","category":"function"},{"location":"api/#Measurements","page":"List of functions","title":"Measurements","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"sic_povm\ntest_sic\ndilate_povm\npovm\nmub\ntest_mub","category":"page"},{"location":"api/#Ket.sic_povm","page":"List of functions","title":"Ket.sic_povm","text":"sic_povm([T=ComplexF64,] d::Integer)\n\nConstructs a vector of d² vectors |vᵢ⟩ such that |vᵢ⟩⟨vᵢ| forms a SIC-POVM of dimension d. This construction is based on the Weyl-Heisenberg fiducial.\n\nReference: Appleby, Yadsan-Appleby, Zauner, arXiv:1209.1813\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.test_sic","page":"List of functions","title":"Ket.test_sic","text":"test_sic(vecs)\n\nTests whether vecs is a vector of d² vectors |vᵢ⟩ such that |vᵢ⟩⟨vᵢ| forms a SIC-POVM of dimension d.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.dilate_povm","page":"List of functions","title":"Ket.dilate_povm","text":"dilate_povm(vecs::Vector{Vector{T}})\n\nDoes the Naimark dilation of a rank-1 POVM given as a vector of vectors. This is the minimal dilation.\n\n\n\n\n\ndilate_povm(E::Vector{<:AbstractMatrix})\n\nDoes the Naimark dilation of a POVM given as a vector of matrices. This always works, but is wasteful if the POVM elements are not full rank.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.povm","page":"List of functions","title":"Ket.povm","text":"povm(B::Vector{<:AbstractMatrix{T}})\n\nCreates a set of (projective) measurements from a set of bases given as unitary matrices.\n\n\n\n\n\npovm(A::Array{T, 4}, n::Vector{Int64})\n\nConverts a set of measurements in the common tensor format into a matrix of matrices. The second argument is fixed by the size of A but can also contain custom number of outcomes.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.mub","page":"List of functions","title":"Ket.mub","text":"mub([T=ComplexF64,] d::Integer)\n\nConstruction of the standard complete set of MUBs. The output contains 1+minᵢ pᵢ^rᵢ bases, where d = ∏ᵢ pᵢ^rᵢ.\n\nReference: Durt, Englert, Bengtsson, Życzkowski, arXiv:1004.3348.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.test_mub","page":"List of functions","title":"Ket.test_mub","text":"Check whether the input is indeed mutually unbiased\n\n\n\n\n\n","category":"function"},{"location":"api/#Nonlocality","page":"List of functions","title":"Nonlocality","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"chsh\ncglmp\nlocal_bound\ntsirelson_bound\ncorrelation_tensor\nprobability_tensor\nfp2cg","category":"page"},{"location":"api/#Ket.chsh","page":"List of functions","title":"Ket.chsh","text":"chsh([T=Float64,] d::Integer = 2)\n\nCHSH-d nonlocal game in full probability notation. If T is an integer type the game is unnormalized.\n\nReference: Buhrman and Massar, arXiv:quant-ph/0409066.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.cglmp","page":"List of functions","title":"Ket.cglmp","text":"cglmp([T=Float64,] d::Integer)\n\nCGLMP nonlocal game in full probability notation. If T is an integer type the game is unnormalized.\n\nReferences: arXiv:quant-ph/0106024 for the original game, and arXiv:2005.13418 for the form presented here.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.local_bound","page":"List of functions","title":"Ket.local_bound","text":"local_bound(G::Array{T,4})\n\nComputes the local bound of a bipartite Bell functional G, written in full probability notation as a 4-dimensional array.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.tsirelson_bound","page":"List of functions","title":"Ket.tsirelson_bound","text":"tsirelson_bound(CG::Matrix, scenario::Vector, level::Integer)\n\nUpper bounds the Tsirelson bound of a bipartite Bell funcional game CG, written in Collins-Gisin notation. scenario is vector detailing the number of inputs and outputs, in the order [oa, ob, ia, ib]. level is an integer determining the level of the NPA hierarchy.\n\nThis function requires Moment. It is only available if you first do \"import MATLAB\" or \"using MATLAB\".\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.correlation_tensor","page":"List of functions","title":"Ket.correlation_tensor","text":"correlation_tensor(p::AbstractArray{T, N2}; marg::Bool = true)\n\nApplies N sets of measurements onto a state rho to form a probability array. Convert a 2x...x2xmx...xm probability array into\n\na mx...xm correlation array (no marginals)\na (m+1)x...x(m+1) correlation array (marginals).\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.probability_tensor","page":"List of functions","title":"Ket.probability_tensor","text":"probability_tensor(rho::LA.Hermitian, all_Aax::Vector{Measurement}...)\n\nApplies N sets of measurements onto a state rho to form a probability array.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.fp2cg","page":"List of functions","title":"Ket.fp2cg","text":"fp2cg(V::Array{T,4}) where {T <: Real}\n\nTakes a bipartite Bell functional V in full probability notation and transforms it to Collins-Gisin notation.\n\n\n\n\n\n","category":"function"},{"location":"api/#Norms","page":"List of functions","title":"Norms","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"trace_norm\nkyfan_norm\nschatten_norm","category":"page"},{"location":"api/#Ket.trace_norm","page":"List of functions","title":"Ket.trace_norm","text":"trace_norm(X::AbstractMatrix)\n\nComputes trace norm of matrix X.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.kyfan_norm","page":"List of functions","title":"Ket.kyfan_norm","text":"kyfan_norm(X::AbstractMatrix, k::Int, p::Real = 2)\n\nComputes Ky-Fan (k,p) norm of matrix X.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.schatten_norm","page":"List of functions","title":"Ket.schatten_norm","text":"schatten_norm(X::AbstractMatrix, p::Real)\n\nComputes Schatten p-norm of matrix X.\n\n\n\n\n\n","category":"function"},{"location":"api/#Random","page":"List of functions","title":"Random","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"random_state\nrandom_state_vector\nrandom_unitary\nrandom_povm\nrandom_probability","category":"page"},{"location":"api/#Ket.random_state","page":"List of functions","title":"Ket.random_state","text":"random_state([T=ComplexF64,] d::Integer, k::Integer = d)\n\nProduces a uniformly distributed random quantum state in dimension d with rank k.\n\nReference: Życzkowski and Sommers, arXiv:quant-ph/0012101.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.random_state_vector","page":"List of functions","title":"Ket.random_state_vector","text":"random_state_vector([T=ComplexF64,] d::Integer)\n\nProduces a Haar-random quantum state vector in dimension d.\n\nReference: Życzkowski and Sommers, arXiv:quant-ph/0012101.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.random_unitary","page":"List of functions","title":"Ket.random_unitary","text":"random_unitary([T=ComplexF64,] d::Integer)\n\nProduces a Haar-random unitary matrix in dimension d. If T is a real type the output is instead a Haar-random (real) orthogonal matrix.\n\nReference: Mezzadri, arXiv:math-ph/0609050.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.random_povm","page":"List of functions","title":"Ket.random_povm","text":"random_povm([T=ComplexF64,] d::Integer, n::Integer, r::Integer)\n\nProduces a random POVM of dimension d with n outcomes and rank min(k, d).\n\nReference: Heinosaari et al., arXiv:1902.04751.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.random_probability","page":"List of functions","title":"Ket.random_probability","text":"random_probability([T=Float64,] d::Integer)\n\nProduces a random probability vector of dimension d uniformly distributed on the simplex.\n\nReference: Dirichlet distribution\n\n\n\n\n\n","category":"function"},{"location":"api/#States","page":"List of functions","title":"States","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"phiplus\npsiminus\nisotropic\nanti_isotropic","category":"page"},{"location":"api/#Ket.phiplus","page":"List of functions","title":"Ket.phiplus","text":"phiplus([T=ComplexF64,] d::Integer = 2)\n\nProduces the maximally entangled state Φ⁺ of local dimension d.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.psiminus","page":"List of functions","title":"Ket.psiminus","text":"psiminus([T=ComplexF64,] d::Integer = 2)\n\nProduces the maximally entangled state ψ⁻ of local dimension d.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.isotropic","page":"List of functions","title":"Ket.isotropic","text":"isotropic(v::Real, d::Integer = 2)\n\nProduces the isotropic state of local dimension d with visibility v.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket.anti_isotropic","page":"List of functions","title":"Ket.anti_isotropic","text":"anti_isotropic(v::Real, d::Integer = 2)\n\nProduces the anti-isotropic state of local dimension d with visibility v.\n\n\n\n\n\n","category":"function"},{"location":"api/#Internal-functions","page":"List of functions","title":"Internal functions","text":"","category":"section"},{"location":"api/","page":"List of functions","title":"List of functions","text":"Ket._partition\nKet._fiducial_WH\nKet._idx\nKet._tidx","category":"page"},{"location":"api/#Ket._partition","page":"List of functions","title":"Ket._partition","text":"partition(n::Integer, k::Integer)\n\nIf n ≥ k partitions the set 1:n into k parts as equally sized as possible. Otherwise partitions it into n parts of size 1.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket._fiducial_WH","page":"List of functions","title":"Ket._fiducial_WH","text":"_fiducial_WH([T=ComplexF64,] d::Integer)\n\nComputes the fiducial Weyl-Heisenberg vector of dimension d.\n\nReference: Appleby, Yadsan-Appleby, Zauner, arXiv:1209.1813 http://www.gerhardzauner.at/sicfiducials.html\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket._idx","page":"List of functions","title":"Ket._idx","text":"_idx(tidx::Vector, dims::Vector)\n\nConverts a tensor index tidx = [i₁, i₂, ...] with subsystems dimensions dims to a standard index.\n\n\n\n\n\n","category":"function"},{"location":"api/#Ket._tidx","page":"List of functions","title":"Ket._tidx","text":"_tidx(idx::Integer, dims::Vector)\n\nConverts a standard index idx to a tensor index [i₁, i₂, ...] with subsystems dimensions dims.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"EditURL = \"https://github.com/araujoms/Ket.jl/blob/master/README.md\"","category":"page"},{"location":"#Ket.jl","page":"Home","title":"Ket.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Dev)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Toolbox for quantum information, nonlocality, and entanglement.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Highlights are the functions mub and sic_povm, that produce respectively MUBs and SIC-POVMs with arbitrary precision, local_bound that uses a parallelized algorithm to compute the local bound of a Bell inequality, and partial_trace and partial_transpose, that compute the partial trace and partial transpose in a way that can be used for optimization with JuMP. Also worth mentioning are the functions to produce uniformly-distributed random states, unitaries, and POVMs: random_state, random_unitary, random_povm. And the eponymous ket, of course.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For the full list of functions see the documentation.","category":"page"}]
}
