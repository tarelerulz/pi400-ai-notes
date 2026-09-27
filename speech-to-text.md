# Speech-to-text on the Raspberry Pi 400

Three different jobs, three different winners:

| job | winner | why |
|---|---|---|
| Live dictation (a few seconds at a time) | **Moonshine tiny** Q8_0 via [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) | fastest, correct punctuation, no length bugs |
| Long recording → plain text | **Canary-180M-flash** Q4_K_M via transcribe.cpp, in **40 s chunks** | most words right; chunking is mandatory |
| Film/video → subtitles | **Parakeet-TDT-0.6B-v3** int8 (sherpa-onnx) + Silero VAD, cues cut on word timestamps | real timestamps, close to Canary's accuracy |

## 1. Live dictation

Short clips, 3 threads, wall time for a fresh process including model load
(transcribe.cpp runs one process per call). Clips: a 7.9 s TTS sentence, the
3.8 s JFK "ask not" clip, and three 10 s cuts of one person's natural
dictation.

| engine | model | 7.9 s clip | 3.8 s clip | 10 s real speech | notes |
|---|---|---|---|---|---|
| **transcribe.cpp** | **Moonshine tiny Q8_0 (27M, 34 MB)** | **1.41 s** | **0.65 s** | **~1.5 s** | exact on the TTS clip; punctuation and casing included |
| transcribe.cpp | Moonshine base Q8_0 (62M) | 2.97 s | 1.31 s | ~3.3 s | same words as tiny on real speech |
| transcribe.cpp | Parakeet-TDT-CTC-110M Q8_0 | 2.06 s | 1.00 s | ~2.5–3.3 s | spells numbers out ("twenty four") |
| transcribe.cpp | SenseVoice-Small (234M) | 5.05 s | 2.62 s | ~6.0 s | lowercase, no punctuation |
| sherpa-onnx | Moonshine v2 tiny (44 MB) | 3.2 s cold / 1.3 s warm | | | breaks above ~8–9 s (see bugs) |
| moonshine-voice (official SDK) | Moonshine tiny-streaming | 5.8 s cold / 5.3 s warm | | | best punctuation of the ONNX paths, but single-threaded |
| audio.cpp | ABR Niagara 19M / 38M | 3.8 s / 5.3 s | | | lowercase, no punctuation; worse WER on the 28 s clip |
| whisper.cpp | tiny.en + VAD | 4.7–5.9 s per utterance | | | 10–14x slower than Moonshine tiny for ~1 word of accuracy |
| llama.cpp (audio input) | Qwen3-ASR-0.6B | 26 s cold / 7 s warm | | | letter-perfect, but heavy (~1 GB) |
| sherpa-onnx | Parakeet-TDT-0.6B-v3 int8 | ~16 s cold / ~7 s warm | | | good text, drops the tail of long clips (see bugs) |
| sherpa-onnx | Zipformer streaming (standard) | 16 s cold / 4.6 s warm | | | ALL CAPS, no punctuation |
| sherpa-onnx | Zipformer streaming 20M | | | | drops the first 1.5–2 s of every utterance |
| — | Nemotron 3.5 ASR streaming 0.6B | | | | slower than realtime |
| VibeASR.cpp | VibeVoice-ASR-BitNet | RTF 6–8 | | | crashes on clips over ~10 s |

On one person's natural speech, Moonshine tiny, base, Parakeet-110M and
SenseVoice produced essentially the same words. **Nothing tested beat Moonshine
tiny for dictation on this CPU**; every alternative was 1.6–4x slower with no
accuracy gain.

### Teaching Moonshine names it keeps mishearing

transcribe.cpp's Moonshine v1 decoder was patched locally to add a logit bias
for chosen subword tokens (for a personal name and "Gentoo"). Lessons from
getting it working:
- **Find the real divergence token from a saved recording**, not from someone
  remembering what the transcript said. The misheard name went wrong at the
  *first* subword, so boosting the second did nothing.
- **Each token needs its own strength.** One shared strength was either too
  weak to win for one word or strong enough to cause loops for another, so the
  config became `token_id:strength` pairs.
- **Boosting a common token (like "▁T") can loop.** On noise-only audio it
  produced repeated words until the token limit. Suppressing a boosted token
  after it has fired once helps, but doesn't fully prevent loops.
- The shipped version adds gating on the decoder's own confidence plus a
  bounded retry, and was re-checked against a set of adversarial clips (noise,
  unrelated speech). The patch isn't upstream.

## 2. Long recordings → text

A 120 s single-speaker dictation recording, 3 threads:

| arm | wall time | result |
|---|---|---|
| **Canary Q4_K_M, fixed 40 s chunks** | **58.5 s (0.49x RT)** | full content; the only arm that got a hard phrase right |
| Canary Q8_0 via audio.cpp (auto 40 s chunks) | 82.5 s | same words, 1.4x slower, 249 MB vs 139 MB |
| Canary, whole file in one call | 72–79 s | **silently skips the middle ⅔** (see bugs) |
| Canary, 20 s chunks | 49.6 s | 2 of 6 chunks came back empty |
| Moonshine tiny, fixed 7 s chunks | 24.0 s | full content, choppy at cuts |
| Moonshine tiny + PulseVAD, merged to ≤25 s | 41.9 s | best Moonshine text |
| Moonshine tiny, whole file | 61.8 s | fails: degenerates into repetition |
| whisper tiny.en | 70.7 s | same mistakes as Moonshine tiny, in the same places |

Canary's published WER is flat across its quantizations (Q4_K_M 1.93% vs F32
1.94% on LibriSpeech clean), so the 139 MB Q4_K_M costs nothing.

## 3. Subtitles, and real word-error rates

Subtitles need timestamps. **Moonshine and Canary don't expose them** in
transcribe.cpp (`--timestamps` returns "unsupported"). whisper.cpp has native
`-osrt`; Parakeet has per-token timestamps in sherpa-onnx.

**Ground truth:** the official closed captions of a 140-minute feature film,
with sound tags, speaker labels and markup stripped before scoring.

> **Trap:** the caption file was **5.32 s early** relative to the audio —
> constant across the whole film, found by cross-correlating the caption on/off
> pattern with the audio's loudness envelope. "Captions end when the audio
> ends" is not enough of a check. Before correcting it, every WER was
> inflated (Canary 22.9% → 9.7% on the same window), and an apparent "Canary
> drops the last seconds of a clip" finding turned out to be this offset.

After correcting for the offset:

**Five 120 s dialogue-dense windows (1,328 reference words):**

| engine | words right | WER | speed (Pi, 3 threads) |
|---|---|---|---|
| Canary Q4_K_M, 40 s chunks | **87.6%** | 14.5% | 0.63x RT |
| Parakeet v3 int8, VAD threshold 0.3 | 86.7% | — | ~0.6x RT |
| Parakeet v3 int8, VAD threshold 0.5 | 84.8% | 17.1% | 0.60x RT |

On one window, whisper base.en scored 8.2% WER vs Canary's 9.7%, but at ~1.2x
realtime (a feature film takes ~3 h on the Pi). whisper small.en runs at
~4.2x realtime (~10 h per film) — not worth it here.

**The whole film with Parakeet** (on the phone VM, 2 threads, ~11–14x
realtime), sweeping the VAD threshold:

| VAD threshold | words right | caption lines with no subtitle | cue starts within 0.5 s |
|---|---|---|---|
| 0.3 | 69.8% | 298 | 80% |
| 0.2 | 72.7% | 268 | 79% |
| **0.1** | **75.0%** | 209 | 78% |
| 0.05 | 75.7% | 202 | 78% |

Lower thresholds recover lines spoken under music, and insertions *fell* as
the threshold dropped. The curve flattens at 0.1. Most remaining misses are
one- or two-word lines ("Mm.", "Whoa!", "No.").

The Orukeet v0.1.0 fine-tune of Parakeet scored 75.6% (+0.6 points) under the
same settings. Neither engine got the film's invented fantasy names right.

**Forced alignment:** CrispASR's 392 MB Canary CTC aligner put 94% of cue
starts within 0.5 s on one 120 s window, but adds ~1.3–1.5x realtime after
Canary. Not yet better than Parakeet for a whole film.

## Bugs and traps (details in [lessons.md](lessons.md))

- **Canary silently skips the middle of long input.** Given 120 s in one call
  it returns fluent text with the correct final sentence, but only ~⅓ of the
  words (5-gram overlap per 40 s third: 78%, 5%, 10%). No warning, exit code 0.
  An upstream decode-budget fix left the output identical. Chunk at 40 s.
- **sherpa-onnx 1.13.4's Moonshine v2 wrapper** returns an empty transcript
  above ~9 s (ONNX broadcast error in cross-attention).
- **Parakeet v3 in sherpa-onnx dropped the last ~18 words** of a 28 s clip, no
  error. Padding with silence didn't fix it.
- **A use-after-free in a subtitle script:** reading `vad.front.samples` after
  `vad.pop()` returns freed memory. Symptom: random runs decode garbage (2,760
  `<unk>` tokens in one window). Copy the samples before popping.
- **Community ONNX exports** with the right filenames can still have the wrong
  graph contract: a Moonshine-medium export loaded fine and then output "The The
  The…". Test with a real decode, not just a successful load.
- **Synthetic TTS audio is a bad test input** for ASR. Low-quality TTS made
  Qwen3-ASR hallucinate; real microphone audio didn't.
- **A smaller file isn't always faster:** a 27 MB int8 Moonshine tiny was both
  slower and less accurate than the 44 MB version.
