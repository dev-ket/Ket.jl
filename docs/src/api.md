# List of functions

```@docs
Ket
```

## Basic

```@docs
ket
ketbra
proj
shift
clock
shiftclock
pauli
gellmann
gellmann!
partial_trace
partial_transpose
permute_systems
cleanup!
symmetric_projector
symmetric_isometry
orthonormal_range
permutation_matrix
n_body_basis
```

## Entropy

```@docs
entropy
binary_entropy
relative_entropy
binary_relative_entropy
conditional_entropy
```

## Entanglement

```@docs
schmidt_decomposition
entanglement_entropy
entanglement_robustness
schmidt_number
ppt_mixture
```

## Measurements

```@docs
Measurement
sic_povm
test_sic
test_povm
dilate_povm
povm
tensor_to_povm
povm_to_tensor
mub
test_mub
```

## Incompatibility

```@docs
incompatibility_robustness
```

## Nonlocality

```@docs
chsh
cglmp
inn22
gyni
local_bound
tsirelson_bound
seesaw
tensor_probability
tensor_collinsgisin
tensor_correlation
nonlocality_robustness
```

## Norms

```@docs
trace_norm
kyfan_norm
schatten_norm
diamond_norm
```

## Random

```@docs
random_state
random_state_ket
random_unitary
random_isometry
random_povm
random_probability
```

## States

```@docs
state_bell_ket
state_bell
state_phiplus_ket
state_phiplus
state_psiminus_ket
state_psiminus
state_supersinglet_ket
state_supersinglet
state_ghz_ket
state_ghz
state_w_ket
state_w
state_dicke_ket
state_dicke
state_horodecki33
state_horodecki24
state_grid
state_crosshatch
white_noise
white_noise!
```

## Channels

```@docs
applykraus
applykraus!
choi
channel_bit_flip
channel_phase_damping
channel_phase_flip
channel_amplitude_damping
channel_amplitude_damping_generalized
channel_bit_phase_flip
channel_depolarizing
```

## Internal functions

```@docs
Ket._dps_constraints!
Ket._inner_dps_constraints!
Ket._partition
Ket._fiducial_WH
```
