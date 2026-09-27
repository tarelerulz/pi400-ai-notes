# Lessons, traps, and bugs found

## How to test on a small board without fooling yourself

- **Time it on your own hardware.** Model-card speeds usually come from Apple
  M-series, x86 with AVX-512, or GPUs. On the Pi 400, measured results ranged
  from "a bit slower than claimed" to ~130x slower (TinyTTS).
- **Check the CPU features before believing an ARM speedup.** `grep Features
  /proc/cpuinfo`. Most new llama.cpp/ONNX ARM work needs `asimddp` (dotprod)
  or `i8mm`; the Pi 4/400 has neither.
- **Leave a core free.** Benchmarks on all 4 cores could freeze the machine;
  3 threads also turned out to be the fastest setting for LLM generation.
- **Use real input.** Low-quality synthetic TTS made an ASR model hallucinate
  where real microphone audio didn't. For ASR accuracy, a 7.9 s TTS sentence
  was too easy to tell good engines from great ones; a 28 s four-speaker clip
  and film audio were not.
- **Save the actual failing input.** Debugging from someone's memory of a
  transcript ("it wrote Taryl") pointed at the wrong token; the saved WAV
  showed it went wrong one token earlier.
- **Check your reference data.** A film's closed-caption file was 5.32 s out
  of sync with the audio. That shift inflated every error rate and created a
  fake "model drops the end of each clip" finding. Cross-correlating caption
  timing with the audio's loudness envelope found it.
- **A narrow test can look clean when the thing is broken.** Two tool-calling
  models aced a weather-city test and then failed a wider adversarial set
  ("what's the capital of France" → a weather call).
- **Loading isn't working.** Community ONNX exports with the right filenames
  loaded without errors and then output garbage. Always check a real decode.
- **Smaller isn't automatically faster.** A 27 MB int8 Moonshine was slower
  and less accurate than the 44 MB version; int8 Kokoro distills were no
  faster than fp32; a 4-bit ternary Parakeet was slower than int8.
- **Architecture predicts speed better than parameter count** (see
  [text-to-speech.md](text-to-speech.md)).
- **Measure more than once, in both orders,** and watch for background load.
  One dictation benchmark ran while an emulator was using ~120% CPU; its
  absolute times are flagged as inflated.
- **Things verified once drift.** Several "working" parts of this setup later
  stopped working unnoticed: a wake-word listener died when a USB headset was
  unplugged, and swap failed to come up because of a boot-order race. Re-check
  after reboots and hardware changes.
- **Watch the disk on a small SD/SSD.** A pip install failed partway through a
  test when the root disk hit 96% full.

## Bugs found in other people's software

Found and reproduced locally. Unless noted, these weren't checked against the
projects' issue trackers.

| project | bug | workaround |
|---|---|---|
| audio.cpp / ggml (Tensor G4 native build) | Native SVE/i8mm/fp16 build computes the sanoTTS heart model wrongly: distorted audio of the wrong length | Build plain ARMv8 (`-DGGML_CPU_ARM_ARCH=armv8-a`); same speed, bit-accurate |
| VibeASR.cpp | CMake checks that the *compiler* accepts `+dotprod`, not that the *CPU* supports it, then forces it on → SIGILL on the A72 | Remove the forced fallback; NEON paths already exist |
| VibeASR.cpp | Crashes (`posix_memalign` failure, `ggml_abort`) on audio longer than ~10 s | none |
| sherpa-onnx 1.13.4 | `OfflineRecognizer.from_moonshine_v2` returns an empty transcript above ~9 s (ONNX broadcast error in cross-attention) | 7 s chunks, or use transcribe.cpp |
| sherpa-onnx 1.13.4 + Parakeet-TDT-0.6B-v3 | Silently drops the final sentence of a 28 s clip | Keep segments short (VAD) |
| Zipformer streaming 20M (en-2023-02-17) | Drops the first 1.5–2 s of every utterance, including on its own test WAV | Use the standard-size model |
| Canary-180M-flash (model behaviour) | On ~120 s input, returns fluent text that skips the middle ⅔; exit 0, no warning | Always chunk at 40 s |
| moonshine-voice SDK | Runs ONNX Runtime single-threaded with no thread option → 4x slower than the same model elsewhere | Use transcribe.cpp |
| bitnet.cpp (July 2026 submodule) | i2_s tensors mislabelled as Q1_0 at load → falls off the fast path (0.8 / 0.7 tok/s instead of 11.8 / 2.46) | Pin the previous submodule commit |
| llama.cpp | `LLM_FFN_RELU_SQR` omits the gate multiply, so reusing it for official BitNet-2B's `relu(gate)² x up` FFN would be silently wrong | Build the FFN explicitly (local patch) |
| llama.cpp `llama-cli` | With stdin from `/dev/null`, loops forever re-running inference even with `--no-conversation` | Use `llama-bench` / `llama-server` |
| Moondream Photon (kestrel) | parakeet-redux on ARM CPU outputs fragments (5.9% of words right); kernels are an encrypted payload | Community sherpa-onnx port works |
| Needle 2 (Cactus) | "roll a d20" → `sides=2`, every time | Fixed in Needle 3 |

## A pattern to avoid in sherpa-onnx VAD code

```python
seg = vad.front
vad.pop()
audio = seg.samples      # WRONG: pop() freed the buffer
```

Read `seg.samples` and `seg.start` **before** `vad.pop()`. The broken version
works on some runs and decodes garbage on others (thousands of `<unk>` tokens),
which makes it look like a model problem.
