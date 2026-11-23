# MLA RoPE numerical diagnosis

The 4K difference starts in the FP32 sum across heads in the shared-key
backward. The compiled PyTorch reduction and the generalized Triton unpack
group the adds differently. Both apply the same FP32 inverse rotation and
round once to BF16. Feeding the two captured sums through the same explicit
FP32 FMA formula and BF16 cast reconstructs both gradients bitwise and
reproduces the complete disagreement mask.

| BF16 replay shape | FP32 head-sum differences | Shared rotary key gradient differences | Max absolute gradient difference | Relative gradient L2 |
|---|---:|---:|---:|---:|
| B2/S4096/H32, 10+10 crop | 52,827 / 524,288 | 7 / 524,288 | 5.684e−14 | 5.879e−8 |
| B2/S8192/H64, scaling fixture | 979 / 1,048,576 | 0 / 1,048,576 | 0 | 0 |

Packed Q and K, the query gradient, and the nonrotary key gradient matched
bitwise at both shapes. All tensors were finite. The fixed-input A/A and
B/B controls matched bitwise. The explicit rotation uses a rounded FP32
imaginary-input product and one FP32 FMA for the real-input product and
addition. Both implementations matched that formula bitwise on the Q and K
forward rotary slices and on inverse Q. A hypothetical BF16 round before
shared-key inverse rotation changed about 166,000 elements at 4K and 338,000
at 8K.

With bitwise-identical packed Q and K, the unchanged compiled MLA tail
with GIFT SAC and cuDNN still produced variation in its query-gradient
output. The confirmed A/A comparison found 10 differences at 4K and 905 at
8K. Tail forward, key gradient, and value gradient matched bitwise. This
observes query-side variation before the RoPE backward receives its
gradient. The tail also includes the sigmoid gate and FP8 output projection,
so this probe alone does not identify a specific cuDNN instruction.

The [harness](../../mla_rope_numerics.py) builds Q, K, and V using the
production MLA projection modules with shared deterministic weights and
captured hidden inputs. It takes identical replay gradients from one GIFT
SAC tail. For the sum intervention it runs the Inductor-generated baseline
reduction and reuses the Triton unpack with identity coefficients and an
FP32 destination. A separate run with the real coefficients and FP32
destination checks the candidate rotation and its final cast.

The confirmed receipts include shapes, strides, finite counts, bit
mismatches, errors, activation markers, and sample intermediate values.
[4K](confirmed/4096.json) · [8K](confirmed/8192.json) ·
[Launch](run.sh) · [Provenance](confirmed/provenance.txt) ·
[4K generated code](confirmed/4096.log) ·
[8K generated code](confirmed/8192.log)

The exploratory run generated the baseline backward modules consumed in
the confirmation and already measured the same fixed-boundary differences.
Its hypothetical late-cast `fma_im` control reused a rounded value from the
early-cast arm. The confirmed receipts reran that formula with the
intended input and include the exact sum intervention.
