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

## A faster ARM decoder for heart and heart-nano

The part of heart / heart-nano that turns the planned speech into sound (the
"decoder") was rewritten by hand for ARM CPUs and offered to audio.cpp as
[pull request #860](https://github.com/0xShug0/audio.cpp/pull/860) (not merged
yet). Same model files, same voice: on 32 test runs, 99.6–99.8% of audio
samples came out identical and the rest differed by the smallest possible step.
The code was written with an AI coding assistant (Claude) and measured on the
hardware below.

**Using it:** it's an opt-in setting, off by default.

```
audiocpp_cli --family sanotts --task tts --model <heart-nano package> ... \
  --session-option sanotts.cpu_decoder=neon
```

Until the PR is merged, build audio.cpp from the
[`sanotts-neon-decoder` branch](https://github.com/tarelerulz/audio.cpp/tree/sanotts-neon-decoder).
Model packages converted before the option existed reject it as unknown; add
`--model-spec-override <audio.cpp>/model_specs/sanotts.json` or re-download the
package. Any 64-bit ARM CPU (Pi 3/4/5/400, phones, Apple Silicon, ARM servers)
can run it; other CPUs keep using the normal decoder.

**Pi 400**, 3 threads, whole program, best of 3:

| voice | text (audio length) | normal | NEON | speed-up |
|---|---|---:|---:|---:|
| heart-nano | 6-sentence email (13.8 s) | 250 ms | 162 ms | 1.54x |
| heart-nano | 4.9 kB text (5.6 min) | 4.94 s | 2.67 s | 1.85x |
| heart | 6-sentence email (14.3 s) | 714 ms | 450 ms | 1.59x |
| heart | 4.9 kB text (5.7 min) | 15.9 s | 9.1 s | 1.75x |

**Pixel 9 phone VM**, 1 thread (faster than 2 inside the VM, for both
decoders): making the audio itself ran at 190x → **577x** realtime for
heart-nano and 50x → 139x for heart on the 4.9 kB text. The whole Wikipedia
"Apollo 11" article (13,185 words, **92 minutes** of speech) took **14.2 s** start
to finish with the NEON decoder against 34.1 s with the normal one, about
9 seconds per hour of speech.

Why it's faster, in short:
1. The weights are rearranged once so the CPU reads them in one straight line
   instead of jumping around memory.
2. Each step works on 4 output channels × 16 time frames at once, filling the
   CPU's NEON vector registers.
3. All threads share every layer instead of splitting the work up by layer.
4. Two consecutive layers with nothing between them are merged into one.

Two things learned along the way:
- **How threads wait for each other matters.** With a sleep-and-wake-up wait
  between layers, 2 threads were *slower* than 1 inside the phone VM (waking
  a thread is expensive in a VM). A short busy-wait fixed it and was also
  slightly faster on the Pi.
- **Long texts need memory, regardless of decoder.** audio.cpp keeps the whole
  result until it writes the file: about 12 MB per minute of speech (1.15 GB
  for the 92-minute article). On a 4 GB Pi that's roughly 3 hours per run.

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
- **Highlighting each word as it's read.** heart's duration model knows when each *spoken* word starts
  and ends (a small local patch makes audio.cpp's CLI write them out). Those spoken words then have to be
  matched to the *written* ones, and two things break a simple match: the phonemizer **joins** short
  words into one spoken word ("was a" → *wʌzə*, "on the" → *ɔnðə*), and **abbreviations are spelled
  out** ("CPU" is 3 letters, 6 sounds). A match that can only give a written word *no* sound zeroed a
  nearby word and ran the highlight one word ahead for the rest of the sentence. Allowing two written
  words to share one spoken word (time split by expected length) and counting all-caps words as letter
  names cut zero-time words from 5 to 1 in 276 (the one left is a "-", which really has no sound).
  For the playback position, mpv's `time-pos` is reliable here: it doesn't start counting before the
  first audio arrives, and it stops while the next sentence is still being synthesised.

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
