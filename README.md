# MSD_musicnn_optimized_fp16.pt

Pre-trained **musicnn** (Pons & Serra) weights for music auto-tagging on the Million Song Dataset (MSD, 50 tags), stored as a PyTorch `state_dict` with half-precision parameters and single-precision batch-norm statistics.

| Property | Value |
|---|---|
| File | `MSD_musicnn_optimized_fp16.pt` (1,617,541 bytes) |
| SHA-256 | `17f79e34e075c2b9b94d394b5f1d82819db28e8887a7e2a6d0618e33540d81db` |
| Container | `torch.save` zip archive (`data.pkl` + `data/N` storages) of an `OrderedDict` |
| Variant | standard (not `_big`): 64 mid-end filters, 200 back-end units |
| Output | 50 MSD tag logits (multi-label; apply element-wise sigmoid) |
| Tensor elements | 792,090 total = 787,159 learnable + 4,920 BN running statistics + 11 `num_batches_tracked` scalars |
| Companion code | `musicnn_msd.py` (module + loader), `check_keys.py` (torch-free structural check) |

> **Verification status (read this first).** Key names, shapes and element counts were verified against the module in `musicnn_msd.py` (75/75 keys, 0 mismatches) without torch. A forward pass and a numerical comparison against a reference implementation have **not** been run. Treat output equivalence to upstream musicnn as unproven until the procedure in [§8](#8-verification-protocol) passes.

---

## 1. Provenance and scope

The architecture is the musicnn family introduced for audio tagging with musically motivated front-end filters, a small mid-end, and a pooled dense back-end. This checkpoint is the MSD-trained, standard-size member. Layer naming (`timbral_*`, `temp_*`, `midend.c*`, `backend.*`) matches the community PyTorch port of musicnn; the checkpoint was exported from that naming scheme and then cast to FP16 on disk ("optimized").

Upstream license terms for the original weights and code apply. Confirm them at the upstream repositories before redistributing; this README does not grant or assert any license.

## 2. Storage format and numerical types

| Tensor class | dtype on disk | Rationale |
|---|---|---|
| Conv / Linear weights and biases | float16 | 2× size reduction |
| BN `weight`, `bias` (affine) | float16 | as above |
| BN `running_mean`, `running_var` | **float32** | variance accumulates small values; FP16 underflow/rounding would bias normalisation |
| BN `num_batches_tracked` | int64 | bookkeeping only, unused at inference |

Consequences:

1. `load_state_dict` copies with dtype conversion, so loading into an FP32 module is exact with respect to the stored FP16 values (the FP16 rounding already happened at export; it is not reversible).
2. `model.half()` converts BN running statistics to FP16 as well, discarding the FP32 stats' precision. If you want mixed precision, keep BN in FP32 (e.g. `model.half()` followed by `for m in model.modules(): if isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d): m.float()` and cast activations at the BN boundary) or use `torch.autocast`.
3. FP16 on disk does not imply FP16 arithmetic at runtime. Quantify the FP32-vs-FP16 drift on your own data before assuming equivalence.

## 3. Architecture (as implied by the tensors)

Notation: `B` batch, `T` time frames, `M = 96` mel bins. Input layout is **time-major**: `(B, T, 96)`, unsqueezed to `(B, 1, T, 96)`.

### 3.1 Front-end (5 parallel branches, 561 features per frame)

| Branch | Kernel (T × F) | Filters | Mechanism |
|---|---|---|---|
| `timbral_1` | 7 × 38 (= ⌊0.4·96⌋) | 204 (= ⌊1.6·128⌋) | pad time by 3+3, valid conv, ReLU, BN, **max over frequency** |
| `timbral_2` | 7 × 67 (= ⌊0.7·96⌋) | 204 | same |
| `temp_1` | 128 × 1 | 51 (= ⌊1.6·32⌋) | `same` padding in time, conv, ReLU, BN, max over frequency |
| `temp_2` | 64 × 1 | 51 | same |
| `temp_3` | 32 × 1 | 51 | same |

Concatenated feature width: 204 + 204 + 51 + 51 + 51 = **561**, which equals the mid-end `c1` kernel width. Timbral kernels span 40%/70% of the frequency axis (pitch-invariant timbre cues at two scales); temporal kernels span the full duration scale at three resolutions with single-bin frequency extent.

Every conv block is ordered **conv → ReLU → BN** (BN eps = 1e-3, momentum = 0.01 in the port). This order is not recoverable from the checkpoint; it was taken from the port's source.

### 3.2 Mid-end (three conv layers, 64 filters each)

Each layer treats the previous feature vector as the frequency axis of a one-channel image and convolves with kernel `7 × W` (W = full width), yielding a single column per time step:

```
x1   = BN(ReLU(conv7×561(front)))
x2   = BN(ReLU(conv7×64(x1)))     ; res2 = x2 + x1
x3   = BN(ReLU(conv7×64(res2)))   ; res3 = x3 + res2
mid  = concat(front, x1, res2, res3)      # 561 + 64 + 64 + 64 = 753
```

Residual additions preserve width, so they are invisible to shape checks; their presence is taken from the port's source. Time is padded by 3 on each side per layer so `T` is preserved.

### 3.3 Back-end

```
mx   = max over T of mid                  # (B, 753)
mean = mean over T of mid                 # (B, 753)
z    = stack([mx, mean], dim=2).flatten(1)  # (B, 1506), INTERLEAVED [mx0, mean0, mx1, mean1, ...]
z    = BN1d(z)                            # backend.bn_in
z    = BN1d(ReLU(Linear 1506→200(z)))     # backend.fc1, backend.bn_fc1
logits = Linear 200→50(z)                 # backend.fc2
```

The interleaved layout is load-bearing: concatenating `[mx | mean]` instead has identical width and loads without error, but permutes the inputs to `bn_in` and `fc1` and silently corrupts predictions. Dropout (p = 0.5 in the port) is inactive in `eval()`.

## 4. State-dict inventory

75 entries. Shapes are `(out, in/groups, kH, kW)` for convolutions.

| Module | Learnable | BN running stats | Notes |
|---|---:|---:|---|
| `bn_input` | 2 | 2 | BN2d over the single input channel |
| `timbral_1` | 54,876 | 408 | conv 204×1×7×38 + bias + BN affine |
| `timbral_2` | 96,288 | 408 | conv 204×1×7×67 + bias + BN affine |
| `temp_1` | 6,681 | 102 | conv 51×1×128×1 |
| `temp_2` | 3,417 | 102 | conv 51×1×64×1 |
| `temp_3` | 1,785 | 102 | conv 51×1×32×1 |
| `midend.c1` | 251,520 | 128 | conv 64×1×7×561 + bias + BN affine |
| `midend.c2` | 28,864 | 128 | conv 64×1×7×64 |
| `midend.c3` | 28,864 | 128 | conv 64×1×7×64 |
| `backend` | 314,862 | 3,412 | `bn_in` (1506), `fc1` 200×1506, `bn_fc1`, `fc2` 50×200 |
| **Total** | **787,159** | **4,920** | plus 11 int64 `num_batches_tracked` = 792,090 elements |

Counts are computed from the enumerated shapes by `check_keys.py`'s expectation table.

## 5. Input contract

Defaults of the original musicnn pipeline (confirm against the port you compare with):

| Setting | Value |
|---|---|
| Audio | 16 kHz mono: this is the resampling target, not a property of the source |
| Source audio | Any codec or bit depth, after decode. As far as I know, MSD training audio was the 30 s lossy MP3 previews, so decoded lossy input is in-distribution. 16-, 24- or 32-bit sources give the same result once resampled to 16 kHz and log-compressed; use a high-quality (sinc) resampler and float32 internally |
| STFT | n_fft 512, hop 256 |
| Mel bins | 96 (the front-end kernels 38/67 are derived from this; other values break the architecture's shape contract) |
| Magnitude → log | `log10(10000·mel + 1)` |
| Patch length | 3 s ≈ 187 frames; longer audio is tiled and logits/probabilities averaged |
| Tensor | `(B, T, 96)`, float |

The model is fully convolutional in `T` up to global max/mean pooling, so other lengths run, but the learned statistics assume ~3 s patches. Train/test length mismatch shifts pooled-feature statistics and therefore `bn_in` inputs.

## 6. Usage

```python
from musicnn_msd import load_msd_musicnn
import torch

model = load_msd_musicnn("MSD_musicnn_optimized_fp16.pt")  # FP32 compute, strict key check
x = torch.randn(1, 187, 96)                                 # replace with real log-mels
with torch.no_grad():
    p = torch.sigmoid(model(x))                             # (1, 50)
```

Structural check without torch:

```bash
python check_keys.py MSD_musicnn_optimized_fp16.pt   # expect: keys expected/found: 75 75 | mismatches: 0
```

## 7. Failure modes that pass `strict=True`

These are shape-compatible and therefore undetectable by loading:

| # | Hazard | Effect |
|---|---|---|
| 1 | conv→BN→ReLU instead of conv→ReLU→BN | BN statistics applied to the wrong activation distribution |
| 2 | `cat([mx, mean])` instead of interleaved stack | permuted `bn_in`/`fc1` inputs |
| 3 | Mid-end without residual adds | different `res2`, `res3` and hence different 753-d features |
| 4 | BN eps 1e-5 vs 1e-3 | small but systematic drift |
| 5 | Wrong mel front-end constants (hop, log compression, normalisation), including a mel computed at 24 kHz and kept at 96 bins, which stretches the filterbank to 0-12 kHz instead of 0-8 kHz, or an `fmax` inherited from the source file's Nyquist instead of being fixed at 8 kHz | distribution shift at `bn_input`; wrong receptive field for every timbral kernel |
| 6 | `.half()` on BN | loss of FP32 running-statistics precision |
| 7 | Treating logits as softmax | tags are independent Bernoulli outputs; use sigmoid |

## 8. Verification protocol

1. **Structural:** `python check_keys.py …` → 0 mismatches. *(done)*
2. **Load:** `load_msd_musicnn(...)` raises on any missing/unexpected key. *(needs torch)*
3. **Reference diff:** feed an identical fixed input to this module and to the upstream port (`oriyonay/musicnn-pytorch`, `trust_remote_code=True`) loaded with matching weights; require `max |Δlogit|` ≲ 1e-3 in FP32. Failures localise by bisecting intermediate tensors (post-front-end, post-mid-end, pooled).
4. **Task-level:** on a labelled MSD split, report ROC-AUC and PR-AUC macro-averaged over 50 tags. The musicnn paper reports ROC-AUC ≈ 0.88 and PR-AUC ≈ 0.29 for this variant in FP32 (verify against the paper's table); compare FP32 vs FP16 paths.
5. **Determinism:** `eval()`, `torch.no_grad()`, fixed input must reproduce logits bit-for-bit on the same device.

## 9. Limitations and responsible use

- Tags reflect MSD/Last.fm social-tag distributions: genre-, era- and Western-popular-music-skewed, noisy, and not a ground truth of musical content. Expect weak performance on non-Western music, speech, and heavily processed or very short audio.
- Probabilities are uncalibrated; thresholds should be tuned per tag on held-out data.
- 3 s receptive tiling ignores long-range musical structure.
- Do not use outputs as evidence about performers or rights holders.

## 10. References

- J. Pons and X. Serra, "musicnn: Pre-trained convolutional neural networks for music audio tagging," late-breaking demo, ISMIR 2019 (arXiv:1909.06654).
- J. Pons, O. Nieto, M. Prockup, E. Schmidt, A. Ehmann, X. Serra, "End-to-end learning for music audio tagging at scale," ISMIR 2018.
- J. Pons and X. Serra, "Designing efficient architectures for modeling temporal features with convolutional neural networks," ICASSP 2017.
- T. Bertin-Mahieux, D. Ellis, B. Whitman, P. Lamere, "The Million Song Dataset," ISMIR 2011.
- Port: `oriyonay/musicnn-pytorch` (Hugging Face).

Citation details above are given from memory of the standard references; verify titles, venues and numbers before citing.
