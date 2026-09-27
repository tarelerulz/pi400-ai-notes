# The same models on a Pixel 9 (Android "Linux Terminal" VM)

Recent Android versions on Pixel phones can run a Debian VM ("Linux Terminal"). It was
used as a second, faster test machine. Important limits:

- **CPU only.** The GPU, the Tensor G4's NPU and NNAPI aren't reachable from
  inside the VM, so "phone NPU" speedups don't apply.
- The VM's cores are a mix of big and little cores. Timings were noisier than
  on the Pi, and 8 threads were *slower* than 4 (0.18x vs 0.12x realtime for
  Parakeet) because the little cores held the big ones back.
- Unlike the Pi 400, the CPU has dotprod, i8mm and SVE2, so it previews what
  those instructions are worth.

## Text-to-speech

One 4-sentence paragraph (~18 s of audio), 3 threads, whole process including
model load:

| voice | phone VM | Pi 400 |
|---|---|---|
| sanoTTS heart-nano | 245–273 ms (~70x realtime) | 334–371 ms (~52x) |
| sanoTTS heart | 352–403 ms (~50x) | 950–976 ms (~20x) |

For an ~11 s email sentence with the model already loaded (generation only):
heart-nano 126–158 ms on the phone vs 172–189 ms on the Pi; heart 204–245 ms
vs 551–629 ms. The bigger model gains more from the faster CPU.

Kokoro-82M (audio.cpp, Q8_0, plain ARMv8 build, 6 threads) ran at RTF
1.6–1.9 on the phone, i.e. slower than realtime, ~50–60x slower than heart. A
listener judged the two "both sound good".

### A compiler bug that garbled the audio

Built with the phone's native instruction set (SVE, i8mm, fp16, bf16),
audio.cpp's heart model produced **audibly distorted speech of a different
length** (333,440 vs 345,216 samples; correlation with the Pi's output 0.01).
Rebuilt for plain ARMv8 (`-DGGML_CPU_ARM_ARCH=armv8-a`, native CPU detection
off), the output matched the Pi's to within 1 LSB, and it was no slower
(RTF 0.019–0.027). The exact kernel responsible wasn't isolated. **Lesson: when
enabling new SIMD paths, compare output against a known-good build, not just
speed.**

## Speech-to-text (dictation)

3 threads, one model at a time, time to transcribe with the model loaded:

| clip | Moonshine tiny (transcribe.cpp) | Parakeet v3 int8 | Orukeet v0.1.0 int8 | Parakeet "redux" ternary (sherpa-onnx) |
|---|---|---|---|---|
| 7.9 s TTS sentence | 560 ms | 466 ms | **391 ms** | 1,292 ms |
| 3.8 s JFK clip | 312 ms | 253 ms | **208 ms** | 709 ms |
| 10 s real dictation (x3) | 598–656 ms | 542–574 ms | **459–478 ms** | 1,544–1,789 ms |

On one person's real dictation, all four produced essentially the same words.
On the phone the 0.6B Parakeet family is fast enough for live dictation; on the
Pi the same model takes ~7 s for an 8 s clip, so Moonshine tiny stays the Pi's
choice.

**Whole-film subtitles:** Parakeet v3 + VAD on the phone ran a 140-minute film
in 10–13 minutes (~11–14x realtime) on 2 threads with `nice`. Four threads
reached ~17x realtime but froze the VM at 86 minutes.

## Ternary / 1-bit models

- **Parakeet "redux"** (a 1.58-bit re-quantization of Parakeet v3) through
  Moondream's own runtime (Photon/kestrel) on CPU got **5.9% of words right**,
  nearly all deletions. It was broken even on NVIDIA's clean sample clip, and
  its CPU kernels are an encrypted binary payload, so it can't be diagnosed. A
  community sherpa-onnx port of the same weights transcribed correctly, but was
  ~3x slower than the int8 model on the phone, and slower than int8 on the Pi
  too.
- **1-bit Bonsai (Q1_0) with llama.cpp's new ARM repack** — prompt/generate
  tok/s, 3 threads:

  | model / build | repack on | repack off |
  |---|---|---|
  | Bonsai-1.7B, dotprod+i8mm build | **81.3 / 19.2** | 11.9 / 8.9 |
  | Bonsai-1.7B, plain ARMv8 build | 14.4 / 10.3 | 14.1 / 10.3 |
  | Bonsai-8B, dotprod+i8mm build | **14.0 / 4.0** | 2.5 / 2.1 |

  Repacking gives ~6–7x prompt processing and ~2x generation, but only when
  dotprod/i8mm are compiled in. **Trap:** `-DGGML_NATIVE=ON` inside the VM
  silently built plain ARMv8 (gcc's `-mcpu=native` didn't pick up the
  features), so the repack was never compiled; the fix was
  `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+i8mm+fp16`. Check
  llama.cpp's `system_info` line for `DOTPROD = 1` before believing a result.
  The Pi 400 can't use this repack at all; its ternary Bonsai-1.7B generates
  ~3–3.7 tok/s.
