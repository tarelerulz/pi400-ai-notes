# Why your CPU matters for AI (and what the Pi 4/400 is missing)

Running AI models on a CPU is mostly one job repeated billions of times:
**multiply pairs of numbers and add the results up.** Compressed ("quantized")
models such as Q4_0 or Q8_0 store those numbers as tiny 4-bit or 8-bit
integers, so the speed question becomes: *how many small-integer
multiply-and-adds can this CPU do per step?*

Newer CPUs have special instructions that do many of them at once. The
Raspberry Pi 4 and Pi 400 (Cortex-A72) have none of them.

## The instructions, in plain words

| instruction | what it does in one step | Pi 4 / Pi 400 |
|---|---|---|
| **NEON** (`asimd`) | basic math on a short row of numbers at once | ✅ the only one it has |
| **dotprod** (`asimddp`) | multiplies 16 pairs of 8-bit numbers and adds them into 4 totals | ❌ |
| **i8mm** (8-bit matrix multiply) | multiplies a small 2x8 by 8x2 block of 8-bit numbers: 32 multiply-adds | ❌ |
| **fp16** (`asimdhp`) / **bf16** | math on half-size decimal numbers, twice as many per step | ❌ |
| **SVE / SVE2** | longer, flexible-length rows of numbers | ❌ |
| **SME** | works on whole tiles of a matrix at once | ❌ |

Without dotprod, the A72 does the same work in several separate steps:
multiply into wider numbers, then add pairs together, then accumulate. That's
why, on the Pi 400, **processing a prompt is only ~2x faster than generating
the answer**, while chips with dotprod/i8mm manage 10–30x.

## Which chips have what

| chip | example devices | adds over the Pi 4/400 |
|---|---|---|
| Cortex-A72 | Raspberry Pi 4, Pi 400 | — (NEON only) |
| Cortex-A76 | Raspberry Pi 5 | dotprod, fp16 (no i8mm) |
| Cortex-X4 / A720 / A520 | Pixel 9 (Tensor G4), other 2024 phones | dotprod, i8mm, fp16, bf16, SVE2 |
| Apple M1 | Macs | dotprod |
| Apple M2 / M3 | Macs | dotprod, i8mm, bf16 |
| Apple M4 | Macs | adds SME |
| Intel / AMD | PCs | their own equivalents: AVX2, then AVX-VNNI / AVX-512 VNNI, AMX on some servers |

These rows are the chip makers' published feature sets. Only the Pi 400 and
Pixel 9 rows were checked on real hardware for these notes.

## What it's worth: a measurement

Same phone, same 1-bit Bonsai model, llama.cpp built with and without dotprod
and i8mm (prompt / generation tokens per second, 3 threads):

| model | dotprod + i8mm on | plain ARMv8 build |
|---|---|---|
| Bonsai-1.7B (Q1_0) | **81.3 / 19.2** | 14.4 / 10.3 |
| Bonsai-8B (Q1_0) | **14.0 / 4.0** | 2.0 / 1.7 |

About **6–7x faster at reading prompts** and ~2x faster at generating, from
instructions alone. (Details in [phone-comparison.md](phone-comparison.md).)

This is also why most "ARM-optimized AI" news doesn't help a Pi 4/400:
llama.cpp's newer ARM speedups (Q4_0/Q4_K/Q1_0 "repacking", KleidiAI kernels)
only switch on when dotprod or i8mm is present. On an A72 they do nothing.

## The other half: memory speed

The instructions mostly speed up **reading your prompt** (lots of math at
once). **Writing the answer** works one token at a time, and each token reads
the whole model from RAM. On the Pi 400 that's limited to about **3.2 GB/s**,
so a 1 GB model can't generate much faster than ~3 tokens per second, whatever
instructions the CPU has. (Measured: `tokens/sec x model size` stays near
3 GB/s across models 9x apart in size; see [llms.md](llms.md).)

So a faster AI computer needs both:
- **new instructions** → faster prompt reading (and faster speech recognition
  and TTS, which are mostly this kind of math)
- **faster memory** → faster answers from LLMs

## Check any computer in 10 seconds

On Linux (including Raspberry Pi OS and Android's Linux Terminal):

```
grep -m1 Features /proc/cpuinfo
```

Look for these words:

| word | means |
|---|---|
| `asimddp` | dotprod |
| `i8mm` | 8-bit matrix multiply |
| `asimdhp` | fp16 math |
| `bf16` | bf16 math |
| `sve`, `sve2` | SVE |
| `sme` | SME |

A Pi 400 prints only `fp asimd evtstrm crc32 cpuid`. If you see only those, most
new AI speedups won't apply to your machine: pick smaller models instead.

## Traps when building for these instructions

- **"Native" builds can silently leave them off.** Inside the Pixel 9's VM,
  llama.cpp's `-DGGML_NATIVE=ON` produced a plain ARMv8 build with no dotprod
  or i8mm. Naming the features explicitly fixed it:
  `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+i8mm+fp16`.
  Check llama.cpp's `system_info` line for `DOTPROD = 1` / `MATMUL_INT8 = 1`.
- **A build can enable an instruction the CPU doesn't have.** One project's
  build script only checked that the *compiler* understood dotprod, then turned
  it on, and crashed on the Pi 400 with "illegal instruction".
- **Faster isn't always correct.** On the Pixel 9, a build using SVE/i8mm/fp16
  produced audibly distorted speech from one TTS model, while the plain build
  was correct and no slower. When you enable new instructions, compare the
  output against a known-good build, not just the speed.
- **Tuning for the old chip doesn't help.** On the Pi 400, building
  llama.cpp for `-mcpu=cortex-a72` was ~6% *slower* than the generic build:
  there were no new instructions to unlock, only scheduling changes.
