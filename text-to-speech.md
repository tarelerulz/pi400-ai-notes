# Text-to-speech on the Raspberry Pi 400

19 engines timed on the Pi 400, plus listening tests. RTF = compute time ÷
audio length; lower is faster, and anything under 1.0 keeps up with live
speech.

Most runs used 3 threads pinned to 3 cores. Two test sentences were used:
- **fox sentence:** "The quick brown fox jumps over the lazy dog. Here is the
  weather for today, sunny with a light breeze and a high of 24 degrees."
- **email sentence:** "Hi, thanks for the update on the voice project. I tested
  the new model this morning and it sounds clear enough for reading email
  aloud. Let me know if you want the numbers." (~11 s of audio)

Engines timed on both sentences agreed within about 10%.

## Results

| engine | params | weights | RTF (Pi 400) | notes |
|---|---|---|---|---|
| **sanoTTS heart-nano** (audio.cpp) | 294k | 1.2 MB | **0.020** | fastest measured; audibly different from heart but intelligible |
| **sanoTTS heart** (audio.cpp) | 2.27M | — | **0.055** | preferred by ear over heart-nano and Piper in a listening test |
| Piper lessac-low, piper1-gpl (warm) | ~15M* | 61 MB | 0.276 | |
| Piper lessac-low, sherpa-onnx (warm) | ~15M* | 61 MB | 0.29–0.31 | clearest word-final consonants of the fast engines |
| Kokoro-7M-Distill (ONNX) | 7.5M | 8.7 MB int8 | 0.29–0.30 | card claims RTF 0.022 on an unnamed 4-thread CPU |
| Paradee-8M v1.0 (ONNX) | 8.1M | 9.0 MB int8 | 0.37–0.39 (1 thread: 0.66) | sounded "about the same" as heart to the listener |
| Piper lessac-low, original binary, fresh process per call | ~15M* | 61 MB | 0.435–0.73 | process start + model load dominate |
| Piper lessac-medium | ~15M* | — | 0.41–0.46 | listener could not tell it from lessac-low |
| Inflect-Nano-v2 (ONNX) | 4.0M | 16 MB | 0.54 | judged more natural than Kitten; output ~40% too quiet |
| Supertonic-2 int8 (sherpa-onnx, 4 threads) | 66M | — | 0.58 | |
| Nix-TTS | 5.2–6.0M | 21 MB | 0.70–0.74 | output ~40% of Piper's loudness |
| Kitten TTS nano 0.8 (fp32) | 14M | 57 MB | 1.03–1.06 | right at realtime |
| Kitten TTS (earlier version) | — | — | 1.47–1.53 | best at releasing word-final stops ("t", "k", "p") |
| TinyTTS | 1.6M | — | ~2.4 | marketed as "53x realtime" |
| Supertonic 3 | ~99M | — | 2.47–2.55 | |
| Pocket TTS | 100M | — | 3.1–4.2 | good voice cloning, far too slow here |
| NeuTTS Nano | — | — | 8.3–8.5 | voice cloning from a 3 s clip |
| MOSS-TTS-Nano | ~100M | — | 18.5–19.5 | |
| Qwen3-TTS-12Hz-1.7B (llama.cpp, Q4_K_M) | 1.7B | 1.04 GB + 446 MB | ~34 | 264 s for 7.7 s of audio |

\* Piper's parameter count isn't published; estimated from the fp32 file size
(bytes ÷ 4). The same arithmetic lands within 2% of Inflect's and Nix-TTS's
published counts.

heart and heart-nano times include starting the program and loading the model
(a one-shot `audiocpp_cli` call). With the model already loaded, generation
alone took 172–189 ms (heart-nano) and 551–629 ms (heart) for ~11 s of audio;
loading costs only ~50–70 ms.

**Kokoro-82M** itself was only measured on the phone VM (RTF 1.6–1.9 via
audio.cpp, plain ARMv8 build) — slower than realtime even there. A listener
rated it and heart "both sound good" with no preference.

## Why architecture decides speed, not size

Parameter count is a poor predictor. Examples from the table:
- Inflect-Nano-v2 (4M params) is ~10x slower than heart (2.3M).
- Kokoro-7M and Paradee-8M are about heart's size but 5–7x slower.
- MOSS-TTS-Nano (~100M) is ~350x slower than heart.

What does predict it:

1. **Non-autoregressive vs autoregressive.** Engines that produce the whole
   clip in one pass (Piper/VITS, Inflect, Nix, heart, Kokoro) are all under
   RTF 1. Engines that generate audio tokens one at a time like an LLM, then
   decode them (Pocket TTS, NeuTTS, MOSS-TTS, Qwen3-TTS), are 3–34x slower
   than realtime on this CPU, regardless of size.
2. **What rate the decoder runs at.** heart's decoder works at the
   mel-spectrogram frame rate (93.75 frames/s at 24 kHz, hop 256), is all
   convolutions (no attention, no LSTM), and does the final inverse STFT as
   plain math. Kokoro-family decoders (StyleTTS2 / iSTFTNet) upsample to 800
   and then 4800 frames/s with residual blocks at each rate, plus an LSTM
   duration predictor and a 12-layer text encoder. Same parameter budget,
   vastly more arithmetic per second of audio.

## Things that aren't speed

- **Clarity vs naturalness.** For reading email aloud, clarity of hard
  consonants mattered more than naturalness. Piper was clearer than heart on a
  paragraph full of word-final stops; heart was preferred overall on normal
  text. Kitten was the only engine that fully released word-final stops, but
  it's slower than realtime.
- **Loudness is a separate bug class.** sherpa-onnx's TTS, Kitten, Inflect,
  Nix-TTS, heart and the Kokoro distills all produce output well below Piper's
  level (heart ~3x quieter). Peak-normalising doesn't fix perceived loudness;
  matching RMS to Piper's average (~0.157 of full scale) with a peak clamp
  does.
- **Piper medium vs low:** 22 kHz "medium" was ~35% slower and a listener
  couldn't tell it apart from 16 kHz "low".
- **Fixing one word's pronunciation in Piper:** espeak-ng accepts inline
  phonemes in double brackets using its ASCII mnemonics, e.g.
  `[[dZ'Entu:]]`. Get the real string from espeak-ng rather than guessing —
  a hand-guessed string produced unrecognisable audio.

## Measured gaps between claims and reality

| engine | claim | measured on the Pi 400 |
|---|---|---|
| TinyTTS | "53x faster than realtime" | 2.4x *slower* than realtime (~130x gap) |
| Kokoro-7M-Distill | RTF 0.022 (4 CPU threads, CPU unnamed) | RTF 0.30 |
| Paradee-8M | ~18x realtime on one thread (Apple M4 Pro) | ~1.5x realtime on one thread |
| int8 versions of the Kokoro distills | smaller, so faster | no faster than fp32 on the A72 |

## Where the tools came from

- sanoTTS heart / heart-nano: run through
  [audio.cpp](https://github.com/0xShug0/audio.cpp)'s `sanotts` family (GGUF);
  needs espeak-ng for phonemes.
- Piper: [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) and
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)'s VITS runtime (sherpa
  was ~34% faster than the original Piper binary).
- Kokoro-7M-Distill: sherpa-onnx bundle; needs misaki phonemes — sherpa's
  built-in espeak front-end mispronounces it.
- Paradee-8M: single ONNX graph, phoneme ids in, 24 kHz waveform out; also
  needs misaki phonemes.
