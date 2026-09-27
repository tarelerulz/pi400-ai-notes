# Local AI on a Raspberry Pi 400 — measured, not guessed

Field notes from about three months (July–September 2026) of running speech
recognition, text-to-speech and small LLMs entirely offline on a Raspberry Pi
400, with a Pixel 9 phone's Linux VM as a faster comparison machine.

Everything here was **run and timed on the actual hardware**. Where a model
card or blog post made a claim, the claim is shown next to the stopwatch
result. Things that were only read about, not run, are labelled as such.

## The hardware

| | Raspberry Pi 400 | Pixel 9 "Linux Terminal" VM |
|---|---|---|
| CPU | 4x Cortex-A72 @ 1.8 GHz (ARMv8.0-A) | Tensor G4, 8 vCPUs exposed to the VM |
| SIMD | NEON only — **no dotprod, no i8mm, no SVE/SME** | NEON + dotprod + i8mm + SVE2 |
| RAM | 3.7 GB usable, measured ~3.2–3.9 GB/s read bandwidth | 6 GB allocated to the VM |
| Accelerators | none (no PCIe, so no AI HAT) | none reachable from inside the VM (CPU only) |
| OS | Gentoo userland on the Raspberry Pi OS 6.12 kernel | Debian (Android Virtualization Framework) |

The Pi 4 has the same CPU and memory system as the Pi 400, so every Pi 400
number should carry over to a Pi 4 with equal RAM.

## The short answers

| job | best choice found on the Pi 400 | speed on the Pi 400 |
|---|---|---|
| Text-to-speech, fast | **sanoTTS "heart"** (2.27M params) via audio.cpp | RTF 0.05 — about 20x faster than realtime |
| Text-to-speech, clearest consonants | Piper `en_US-lessac-low` via sherpa-onnx | RTF 0.29 |
| Live dictation | **Moonshine tiny** (27M, Q8_0 GGUF) via transcribe.cpp | ~1.5 s for an 8 s utterance, cold |
| Batch transcription, most words right | Canary-180M-flash Q4_K_M via transcribe.cpp, **40 s chunks** | ~0.5x realtime |
| Subtitles (needs timestamps) | Parakeet-TDT-0.6B-v3 int8 + Silero VAD, word-timed cues | ~0.6x realtime (Pi), ~14x realtime (phone) |
| General chat LLM | Qwen3.5-2B Q4_0 | ~2.4 tok/s generation |
| Fast small LLM | LFM2.5-1.2B QAD-Q4_0 | ~4.8 tok/s |
| Tool routing ("is this a weather/time/dice question?") | LFM2.5-1.2B with a grammar-constrained one-word answer | 16/18 correct, ~0.7–4 s |

RTF (real-time factor) = seconds of compute per second of audio. Below 1.0 is
faster than realtime.

## What's in here

- [text-to-speech.md](text-to-speech.md) — 19 TTS engines timed, and why
  architecture matters far more than parameter count.
- [speech-to-text.md](speech-to-text.md) — dictation, long-file transcription
  and subtitles, with real word-error rates against closed captions.
- [llms.md](llms.md) — why generation is memory-bandwidth-bound on this chip,
  which quantizations and tricks help (few do), ternary/BitNet, tool routing.
- [phone-comparison.md](phone-comparison.md) — the same models on a Pixel 9's
  Linux VM, and a compiler bug that garbled audio there.
- [lessons.md](lessons.md) — testing method, traps, and bugs found in other
  people's software.

## The three things most worth knowing

1. **On the Pi 400, LLM generation speed is set by RAM bandwidth.**
   `tokens/sec x model size` is nearly constant (~3 GB/s). Smaller model =
   faster; almost nothing else moves it.
2. **Most "ARM-optimized" speedups skip this chip.** They need dotprod or
   i8mm, which the Cortex-A72 doesn't have. Check `/proc/cpuinfo` Features
   before believing a speed claim — the A72 shows only `fp asimd evtstrm crc32`.
3. **Model-card speeds usually come from much faster CPUs.** Measured gaps
   ranged from "a bit slower" to 130x (TinyTTS: claimed 53x faster than
   realtime, measured 2.4x *slower* than realtime). Always time it yourself.

## How things were measured

- Three cores for tests (`taskset -c 1-3` or `0-2`), leaving one free so the
  machine stays responsive. Four threads were occasionally used and are
  marked when they were.
- Nothing else heavy running, unless noted (one run is flagged because an
  emulator was running).
- Several runs per measurement; ranges are shown where runs differed.
- For speech-to-text accuracy: word error rate (WER) against real closed
  captions of a 140-minute feature film, plus clean single-speaker dictation
  and a public 28-second four-speaker test clip.

## Licence

These notes are MIT-licensed (see [LICENSE](LICENSE)). The models and tools
they mention have their own licences.
