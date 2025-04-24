using JuMP
using LinearAlgebra    
import SCS
using Hypatia
using Ket
# Construct the Choi operator of an Amplitude Damping channel 
γ=67/100
# Write a Cell with the Kraus operators
# Since this map is Completely positive, left and right Choi operators are the same.
K0=[1 0; 0 sqrt(1-γ)]
K1 = [0 sqrt(γ); 0 0]
# Declare the Channels which will be used
C1= choi([K0,K1]);

# Construct the Choi operator of an Bit Flip channel 
η=87/100
K0=sqrt(η)* I(2)
K1=sqrt(1-η)*[0 1; 1 0]

C2 = choi([K0,K1])

C = Array{ComplexF64}(undef, 4, 4, 2)  # Création d’un tableau 3D vide (2×2×2)

C[:, :, 1] = C1
C[:, :, 2] = C2

N=size(C,3); #Obtain the number of channels N
k=2; #Set the number of uses k equals 2

d=Int(sqrt(size(C[:,:,1],1)));
dIn=d;
dOut=d;
DIM=[d d d d];
p_i=ones(1,N)/N;
println(d ," ", p_i)

model = Model(SCS.Optimizer)
T = [@variable(model, [1:Int(dIn^(2*k)), 1:Int(dOut^(2*k))] in HermitianPSDCone()) for i in 1:N]
#@constraint(model, partial_trace(sum(T), [1]) == I)
@constraint(model, tr(partial_trace(sum(T), [2])) == dOut ^2)
for j = 1:N
    @constraint(model, tr(sum([T[i]*kron(fill(C[:,:,j],k)...) for i in 1:N])) == 1)
end

@objective(
    model,
    Max,
    sum([real(LinearAlgebra.tr(p_i[i]* T[i] * kron(fill(C[:,:,i],k)...))) for i in 1:N])
)

optimize!(model)
assert_is_solved_and_feasible(model)
solution_summary(model)

Tsolution = [value.(t) for t in T]

display(partial_trace(Tsolution[1],[2]))
println(tr(partial_trace(Tsolution[2],[2])))
