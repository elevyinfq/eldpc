# Relay-BP

This branch adds Relay-BP to the GPU core at base `79ee5e5`; the regenerated core manifests
pin this revision. `relay=0` is the default and preserves the original exactness contract,
verified against the base build and with the patch tools. `relay > 0` selects a new operating
point, requires `decoder=bp`, and has no OSD post-processing. The CPU core refuses the Relay
keys, including when `device="auto"` selects the CPU. Use `device="gpu"` explicitly.

The mode chains memory-BP legs. Leg 0 starts from the priors with uniform memory strength;
later legs start from the preceding leg's final marginals and draw per-column memory strengths
using a SplitMix64 hash of the seed, leg and column. Each leg stops on syndrome convergence or
its iteration cap. The decoder collects up to S converged solutions, returning the one with the
smallest sum of original prior LLRs over its set bits; an exact weight tie keeps the earlier leg.
A cap may stop the search before S solutions: any converged solution is still accepted.

| Key | Default | Accepted range and meaning |
|---|---:|---|
| `relay` | 0 | Integer 0..65535; 0 disables the mode, S > 0 is the number of converged solutions to collect. |
| `relay_legs` | 300 | Integer 1..1000000; total leg cap, **including leg 0**. Caps 5, 10 and 20 are valid. |
| `relay_pre_iter` | 80 | Integer 1..1000000; iteration cap for leg 0. |
| `relay_set_iter` | 60 | Integer 1..1000000; iteration cap for each later leg. |
| `relay_gamma0` | 0.1 | Any finite float; uniform memory strength for leg 0. |
| `relay_gamma_lo` | -0.24 | Any finite float, no greater than `relay_gamma_hi`; lower endpoint of the later-leg uniform interval. |
| `relay_gamma_hi` | 0.66 | Any finite float, no less than `relay_gamma_lo`; upper endpoint of that interval. |
| `relay_seed` | 0 | Integer 0..9000000000000000000; seed for reproducible gamma draws, reset for every decode. |

Every `relay_*` key requires `relay > 0`. The normal config keys `decoder=bp`, `bp_method=ms`
and `max_iter` are required; `max_iter` must be an integer 1..2147483647 but the mode uses its
own iteration caps. `ms_scaling_factor` defaults to 1.0 in this mode; an explicit finite positive
value overrides it. The schedule remains `parallel`. The supplied JSON configurations
[`relay-s1.json`](../configs/relay/relay-s1.json) and
[`relay-s5.json`](../configs/relay/relay-s5.json) set the seed to 20261018.

```python
import numpy as np
import eldpc

h = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
o = np.array([[1, 0, 0]], dtype=np.uint8)
priors = np.full(3, 0.01)
syndrome = np.array([1, 0], dtype=np.uint8)
config = dict(decoder="bp", bp_method="ms", max_iter=1,
              relay=1, relay_legs=5, relay_pre_iter=80, relay_set_iter=60,
              relay_gamma0=0.1, relay_gamma_lo=-0.24, relay_gamma_hi=0.66,
              relay_seed=7, ms_scaling_factor=1.0)
with eldpc.Decoder(h, o, priors, config=config, device="gpu") as decoder:
    correction, observables, status = decoder.decode_full(syndrome)
    if status & eldpc.ST_SYNDROME_UNMATCHED:
        raise RuntimeError("No Relay-BP leg converged within the budget")
    assert np.array_equal((h @ correction) % 2, syndrome)
```

A shot with no converged leg returns the last leg's hard decision with
`ST_SYNDROME_UNMATCHED`; check the status or parity before accepting it. The zero syndrome
returns a zero correction immediately. The status iteration count sums iterations across legs
and saturates at 65535; soft output is the **last leg's** marginals, even when an earlier leg's
correction wins. Library batch decoding is a sequential loop over shots.

The algorithm follows T. Müller et al., [arXiv:2506.01779](https://arxiv.org/abs/2506.01779).
The implementation's gamma stream differs from IBM's `relay_bp`; per-shot parity with that
package is not promised. Its `num_sets` counts subsequent legs, whereas `relay_legs` includes
leg 0. See [NOTICE](../NOTICE) for attribution.

The unchanged patch tools in `tools/relay/` need a development `dbench` adapter to the decoder
ABI; they are not standalone package entry points. `t_relay_config.py`,
`t_relay_determinism.py`, `t_relay_model.py` and `t_relay_unit.py` check mode-on behavior;
the last requires a separate, non-installed `FASTBP_TEST_RELAY_DUMP=1` build. The repository's
`tests/test_bit_identity.py` and `verify/run_verify.py --quick` check the original mode against
an independent reference (and against a base build with `ELDPC_REFERENCE_LIB` set).

## Limitations

All GPU verification and tests for this revision ran on sm_120 (RTX 5080). No runtime tests
ran on sm_89 (RTX 6000 Ada). Relay-on verification includes the AVX-512 host build with the
default GPU core; Relay-on has not been tested with the `ada64` core. The full suite exercised
`ada64` with Relay off on the RTX 5080.

A static CUDA 13.3.73 resource comparison against base `79ee5e5` shows 12 fused BP kernels on
sm_89 growing by 40–104 stack bytes and 7–47 STL/LDL instruction sites. Non-fused kernels also
grow: for example, `bp_kernel<0,0,false,256,…>` grows from 904 to 976 stack bytes and from
235 to 259 STL/LDL sites. This includes kernels used with Relay off. On sm_120, no matched
kernel grows in stack or STL/LDL sites. These are static resource counts; the runtime impact
on Ada, with Relay on or off, is unmeasured. Reproduce with
`python tools/relay/res_diff.py <base-core.so> <relay-core.so> --all`, setting `CUOBJDUMP`
to a wrapper that invokes `cuobjdump -arch sm_89` (or `-arch sm_120`).

Mode-on verification covers round-to-nearest. Directed rounding is not covered by the original
exactness contract for this mode: gamma generation uses round-to-nearest even when other
operations follow the caller's rounding mode. Infinite LLRs (for example endpoint priors) can
produce non-finite memory biases; use finite LLRs for this operating point. Each leg has a host
synchronization, so a leg cap is a work budget, not a latency guarantee.
