# Default subpixel averaging becomes kottke_edge; differentiable companion mode (spec, batch 4)

Status: spec, 2026-10-31. User decision (via the coordinator): "求不求梯度可以用两个版本，kottke edge不支持梯度是ok的。默认确保forward准确"
(gradient and non-gradient work may use two versions; kottke_edge not supporting gradients is fine;
the default must guarantee an accurate forward). Batch 4, after batch 3 lands, so this default change is not
mixed with other movers.

Evidence:
- Sharp-edge set (lq/external-refs-rerun) at ppw 20/30/40:
  - G1-C prism, εr 8: polarized 16/21/19% (does not converge); kottke_edge 18/6.1/5.5%.
  - G1-E rotated boxes (εr 4/12) plus sphere: polarized 21/19/18% (does not converge); kottke_edge 21/11.8/9.3%.
- p16s0: kottke_edge equals the instamax arm, about 1.8× better than polarized.
- Smooth spheres: equal across modes (`docs/assessments/fdtd-kottke-edge-2026-10-31.md`).

## 1. One global default change

- `SubpixelSpec()` defaults change from `samples=(1, 1, 1)` (staircase) to `samples=4, averaging="kottke_edge"`.
  The PEC treatment stays `pec="staircase"`.
- This is one switch for every scene that does not set `subpixel_samples`. There is no switching based on trainability.
- Scenes that set the averaging explicitly are unaffected:
  - `benchmark/scenes/_common.py` uses polarized samples=3;
  - the PCB declared-band inputs use polarized [1,1,512];
  - every test with an explicit `SubpixelSpec`.
  These are confirmed bitwise by the existing torch.equal sets: the 20-case compiler set and the PCB replay gate.
- Batch-level re-pin with an old/new diff report produced by the integration runner. No other movers ride
  in the same batch.

Blast radius (counted on master d03a700b by grep; the integration run gives the exact list):
- 23 benchmark scene files have no explicit subpixel, so they move:
  - antenna/*;
  - array/four_element_linear;
  - coverage/grid_geometry;
  - media/{anisotropic, debye, kerr, modulated, sigma_e_drude}_slab;
  - rf/*;
  - sar/antenna_near_phantom.
  Rows built on `_common.base_scene` do not move. Vacuum and PEC-only scenes compile identically, because
  kottke_edge only changes dielectric and μ interface edges. Expect the rows with axis-aligned dielectric
  slabs to move only where faces are off-grid.
- 405 test files construct a `Scene` without an explicit subpixel. Most are vacuum or PEC and stay
  unchanged. Dielectric pins move and are re-pinned in the batch.
- Bit-identity gates on the default include:
  - the rf_port, port_step_graph and terminal_wave scenes;
  - the existing-monitor runner;
  - the boundary probe;
  - the smooth-surface probe.
  These re-pin through their regeneration paths, with BASE_COMMIT moved to the batch head and staged for the
  cloud. Their scenes are mostly vacuum or PEC, so most should stay torch.equal; the runner reports which actually move.
- Compile cost of the default path: from samples=1 (one point per component) to a 4³ band-only
  supersample around interfaces. On p16s0, prepare goes from about 1 s to 6.0/5.4 s (FDFD/FDTD). Scenes
  without interfaces add only the band test.

## 2. No silent jump for gradients (fail closed)

- If the effective averaging is `kottke_edge` and the geometry is trainable (any geometry tensor
  `requires_grad`, or a SceneModule geometry parameter), preparation raises the existing ValueError.
- The message is reworded to name the alternatives: "kottke_edge (the default subpixel averaging) is not
  differentiable in geometry; set SubpixelSpec(averaging='<differentiable mode>') or 'polarized' for shape
  gradients."
- The mode is never switched automatically. The existing census entry is updated with the new wording and
  the default-path trigger.
- Census of what this blocks: 112 test files use `requires_grad` with a default subpixel. The ones with
  trainable GEOMETRY need an explicit mode, polarized for today's numbers or the new mode once it lands.
  Structure material-value gradients stay allowed. The batch migrates these tests explicitly and lists them;
  examples and docs too.
- Density-gradient follow-up: effective `kottke_edge` also rejects any trainable tensor field of a
  `MaterialRegion`, including density, with: "kottke_edge (the default subpixel averaging) does not
  support gradients through MaterialRegion design densities; set SubpixelSpec(averaging='polarized')
  for density/topology optimization." `kottke_smooth` is not qualified for density gradients.
  Existing density workflows explicitly select `samples=(1,1,1), averaging="arithmetic"` to
  preserve the master default; intentional polarized scenes retain their policy.

## 3. Differentiable companion mode (approved)

**Experiment first** (about 0.5–1 Codex day, about 2 GPU-h, no credits).
- Fixed in every arm: the in-window first-moment normal and the physical-diagonal blend.
- Factorial over {binary occupancy, witwin fractional coverage-ramp fill} × {3³, 4³} samples, on the per-edge
  window of kottke_edge.
- ε is built offline, solved with witwin FDFD at f_eff, which is the same discretisation as FDTD to 0.15%.
- Scored against the existing external references for G1-C, G1-E and p16s0 at ppw 20/30/40 (sharp-edge set)
  and 16/24/32 (p16s).
- The binary-4³ arm must reproduce kottke_edge, as a control.
- Output: per-arm absolute error and order. The question to answer is whether fractional-4³ gets within
  about 10% of binary-4³.

**Then implement the best FRACTIONAL arm** as `SubpixelSpec(averaging="kottke_smooth")` (name proposed) on the
polarized path. Every ingredient is differentiable: fills, the moment of fills, and the blend.

Gates:
- the existing modes stay torch.equal;
- FDFD/FDTD consistency at f_eff ≤ 0.5%;
- shape-gradient FD validation, adjoint vs central FD within 1e-2 relative, on a rotated box and a prism
  (G1-C-like geometry);
- the accuracy table against kottke_edge on G1-C, G1-E and p16s0.

If it lands within about 10% of kottke_edge, it becomes the recommended gradient mode and the mode named in
the §2 message. If not, the message names `polarized`, and the gap is documented.

## ETA

| Step | Codex days | GPU-h |
|---|---:|---:|
| §3 experiment | 0.5–1 | 2 |
| §3 implementation and FD gates | 1.5 | 2–3 |
| §1 + §2 code: default switch, raise wording, census, test and example migration for trainable geometry | 1 | 1 |
| Batch-4 integration re-pin and diff report | runner | runner |

Total: about 3–3.5 Codex days.
