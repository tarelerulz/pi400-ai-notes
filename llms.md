# Small LLMs on the Raspberry Pi 400 (llama.cpp)

## The one fact that explains everything

**Token generation on the Pi 400 is limited by memory bandwidth.** Each
generated token reads the whole model from RAM once, so `tokens/sec x model
size` comes out nearly constant:

| model | weights | generation tok/s | implied bandwidth |
|---|---|---|---|
| LFM2-350M Q4_0 | 207 MB | 14.40 | 2.98 GB/s |
| LFM2-350M Q4_K_M | 216 MB | 14.00 | 3.03 GB/s |
| Qwen3-1.7B Q4_K_M | ~1,223 MB | 3.11 | ~3.8 GB/s |
| qwen2.5-coder-3B Q4_K_M | 1,840 MB | 1.69 | 3.10 GB/s |

Measured directly with a threaded streaming-read test: 3.85 GB/s on 1 thread,
3.60 on 2, 3.20 on 4 (more threads are *worse* because they contend for the
bus). llama.cpp reaches ~97% of that ceiling, so the software isn't the
problem. **Pick the smallest model that's good enough**; nothing else comes
close in effect.

**Prompt processing is compute-bound**, and weak for a different reason: the
A72 has no dotprod or i8mm instructions (`/proc/cpuinfo` Features: `fp asimd
evtstrm crc32`). Prompt processing runs only ~2x faster than generation here,
where chips with dotprod/i8mm manage 10–30x.

## Speeds

`llama-bench`, 3 threads unless marked, prompt 64 / generate 32 tokens.

| model | quant | prompt tok/s | generate tok/s |
|---|---|---|---|
| LFM2.5-230M | QAD-Q4_0 | 49.0 | 21.4 |
| LFM2.5-230M | Q8_0 | 54.9 | 13.3 |
| LFM2-350M | Q4_0 (4 threads) | 39.0 | 14.4 |
| Qwen3.5-0.8B | Q4_K_M (4 threads, pp512/tg128) | 12.9 | 4.46 |
| LFM2.5-1.2B | QAD-Q4_0 | 8.28 | 4.77 |
| LFM2.5-1.2B | Q4_K_M | 7.99 | 4.59 |
| BitNet-b1.58-2B-4T | i2_s, bitnet.cpp (4 threads) | 11.80 | 2.46 |
| BitNet-b1.58-2B-4T | TQ1_0, patched mainline (4 threads) | 7.02 | 2.89 |
| BitNet-b1.58-2B-4T | TQ2_0, patched mainline (4 threads) | 9.18 | 2.80 |
| RWKV-7 G1j 1.5B | Q4_K_M | 6.07 | 3.26 |
| RWKV-7 G1 1.5B | Q4_K_M | 4.93 | 3.05 |
| Ternary-Bonsai-1.7B | Q2_0 g64 | 4.31 | 3.04 (3.72 on 4 threads) |
| Spark-X2.5-1.7B | Q4_K_M (4 threads, pp512/tg128) | 6.39 | 2.99 |
| Qwen3.5-2B | Q4_0 | 5.49 | 2.44 |
| Qwen3.5-2B | Q4_K_M | 5.10 | 2.27 |
| LFM2.5-2.6B | QAD-Q4_0 / Q4_K_M | — | ~2.0 |
| Qwen2.5-Coder-3B | Q4_0 | 3.00 | 1.64 (noisy) |
| Qwen2.5-Coder-3B | Q4_K_M | 2.86 | 1.72 |
| Falcon3-7B-1.58bit | i2_s, bitnet.cpp (older build) | 4.0 | 1.08 |
| Ternary-Bonsai-8B | Q2_0 g64, first load from disk | — | 0.77 |

Notes:
- **RWKV isn't faster than a transformer** at these context sizes; it's the
  same bandwidth limit. Its advantage (flat cost per token at any length) only
  shows in long sessions.
- **Ternary models (BitNet, Bonsai)** help only because they're smaller in
  bytes. TQ1_0 generates faster than i2_s (smaller file) but processes prompts
  40% slower (its unpacking arithmetic costs compute).
- Too big for 3.7 GB: Qwen2.5-Omni-3B with its projector (3.6 GB), MoE models
  like LFM2-8B-A1B (~4.4 GB at Q4). Gemma-4-E2B Q4_0 (3.35 GB) technically fits
  but wasn't benchmarked.

## What was tried to make it faster

| idea | result |
|---|---|
| Build with `-DGGML_NATIVE=ON` (`-mcpu=cortex-a72`) | **~6% slower** prompt processing, same generation. There were no new instructions to unlock, and GCC's A72 scheduling did worse than generic ARMv8. |
| Q4_0 instead of Q4_K_M | +4–8% on 2B-class models; lost tool-calling accuracy on the 230M model (below). *Not* from ARM repacking: llama.cpp's Q4_0 repack paths need dotprod or i8mm (checked in `repack.cpp`); the cause wasn't profiled. |
| Draft-model speculative decoding (0.5B drafting for 3B, 2 tokens ahead) | **1.13x** — the only thing that moved generation. More lookahead was worse. |
| n-gram self-speculation | 0.97x (slower) on normal text; only wins when the answer copies the prompt. |
| LiquidAI DSpark (draft model for LFM2.5-2.6B) | no gain (1.98 → 2.00, then 1.73 tok/s), despite 49–63% draft acceptance. Marketed as up to 3.18x on an H100. |
| Split thread counts: `-t 3 -tb 4` | Free win. Generation peaks at 3 threads (the 4th adds bus contention); prompt processing keeps scaling to 4. |
| New llama.cpp ARM kernels (KleidiAI, Q4_K/Q1_0 repack, tiled k-quant matmul) | Don't apply: they need dotprod/i8mm/SVE, or are x86-only. |
| Run it on the GPU (Vulkan, V3D 4.2) | llama.cpp: **67x slower** generation; prompt processing never got past shader compilation. Hand-written shaders: faster than the CPU for f32 matrix-vector, still 2.3x slower for 4-bit. [Details below](#can-the-pi-400s-gpu-help-not-through-llamacpp-67x-slower-hand-written-shaders-show-what-it-can-do). |

Speculative output was **not bit-identical** to plain greedy decoding, even at
temperature 0: verification runs a batch through a different matmul path and
float addition isn't associative, so near-tied tokens can flip. The output is
deterministic and not worse, just not byte-exact.

## Can the Pi 400's GPU help? Not through llama.cpp (67x slower); hand-written shaders show what it can do

The Pi 400's GPU is a VideoCore VI (V3D 4.2). **VC4CL**, the OpenCL project people point to, only supports
the older VideoCore IV (Pi 1/2/3/Zero), so it isn't an option. The working route is Mesa's **v3dv** Vulkan
driver plus llama.cpp's Vulkan backend, so that's what was measured.

What the GPU offers (`vulkaninfo`, V3DV Mesa 25.0.7, Vulkan 1.3):

| feature | V3D 4.2 | why it matters |
|---|---|---|
| 16-bit float math (`shaderFloat16`) | **no** | fast GPU inference does its math in fp16 |
| 8-bit integer math (`shaderInt8`, int dot) | **no** | the other fast path, for quantized weights |
| 16/8-bit storage | yes | can *store* small numbers, but must compute in fp32 |
| shared memory per workgroup | **16 KB** | small tiles for matrix multiplication |
| memory | 1.85 GB, shared with the CPU | same RAM, same ~3 GB/s as the CPU |

Stock llama.cpp (build 4d756bc72) refuses to start on it: *"Shared memory size too small for matrix
multiplication."* It aborts the whole device if any quantization type doesn't fit in 16 KB; only `iq2_s`
(an 8 KB lookup table) fails. A one-line local patch that disables just that type lets it run.

LFM2.5-1.2B QAD-Q4_0, same binary, 3 threads on 3 pinned cores:

| | CPU (`-ngl 0`) | GPU (`-ngl 99`) |
|---|---|---|
| generation (tg16) | **4.67 tok/s** | **0.07 tok/s** (~14 s per token) |
| prompt processing (pp64) | **7.32 tok/s** | never ran: still compiling one shader after 30 min |
| prompt processing, 1 token at a time (pp16, `-ub 1`) | **3.62 tok/s** | **0.05 tok/s** |
| time before the first token | ~2 s | ~8.5 min (generation shaders) |

With `-ub 1` the GPU reuses the small shaders that generation already compiled, so it can read a prompt, one token at a time, about 72x slower than the CPU doing the same. (`-ub 8` doesn't help: attention still needs a large matrix-multiply shader, which also never finished compiling.)

Most of the start-up time is the driver compiling llama.cpp's shaders on the CPU. The big Q4_0
matrix-multiply shader for prompt processing never finished compiling, and `test-backend-ops` didn't get
past shader creation in 25 minutes either. Mesa's shader cache didn't help, since the compile never completed.

### What the GPU itself can do: hand-written shaders

To separate the hardware from llama.cpp's shaders, the same jobs were run with small hand-written Vulkan
compute shaders, every result checked against the CPU. W = 8192 x 2048 (the size of LFM2.5-1.2B's largest
matrix):

| job | CPU (3 threads) | GPU, best hand-written shader |
|---|---|---|
| read memory | 3.6 GB/s | **6.1 GB/s** (sum verified) |
| f32 multiply-add | — | 12.5 GFLOPS |
| y = W x, f32 weights | ~19–20 ms (plain C and llama.cpp) | **12.0 ms — 1.6x faster than the CPU** |
| y = W x, 4-bit Q4_0-style weights | **~3.2 ms** (llama.cpp, scaled from its 4096 x 14336 case) | 7.5 ms |
| llama.cpp's own Vulkan Q4_0 shader | | 230 ms |

**Correction:** an earlier version of this page said even a perfect GPU couldn't win at generation, because
it shares the CPU's RAM. That was wrong. The GPU's path to RAM is ~1.7x faster than what the A72 cores
reach, and with f32 weights a well-written GPU matrix-vector beats the CPU. With 4-bit weights it still
loses: without int8 or fp16 math it has to unpack every weight in f32, so 4-bit is limited by its
arithmetic, not by memory, while llama.cpp's NEON code unpacks much faster. Since small models only fit
because of 4-bit weights, the CPU remains the better place to run them on this machine.

What made the shaders fast, in order:

1. **256 threads per workgroup.** Every test scaled almost linearly up to it (memory reads: 0.8 GB/s at 32
   threads, 6.1 at 256).
2. **Weights stored row-interleaved:** one thread per output row, and element *j* of every row stored next
   to element *j* of the neighbouring rows, so neighbouring threads always read neighbouring memory and no
   sums across threads are needed. (A workgroup per row, the usual layout, stayed at ~1 GB/s.)
3. **The input vector in shared memory** (8 KB fits in the 16 KB), fetched once per workgroup instead of
   once per weight per thread: f32 3.5 → 5.6 GB/s, 4-bit 13.4 → 7.5 ms.

Splitting rows into parts (split-K) and several rows per workgroup did not help. The driver exposes only
basic subgroup operations (no `subgroupAdd`), so sums across threads have to go through shared memory.
This is 31x faster than llama.cpp's shader for the same 4-bit matrix, but a whole model written this way
would still be estimated at roughly 2 tokens/s for LFM2.5-1.2B against the CPU's 4.7 (an estimate, not
measured).

Testing it didn't need any installs: Raspberry Pi OS ships the v3dv driver, and pointing a Gentoo system's
`VK_ICD_FILENAMES` and `LD_LIBRARY_PATH` at Raspberry Pi OS's `libvulkan.so.1` and `libvulkan_broadcom.so`
(plus `libxshmfence` and `libwayland-client`) was enough for `vulkaninfo` and llama.cpp to find the GPU.

## Practical gotchas

- **Reasoning models return empty answers at small token budgets.** Qwen3.5-2B
  and LFM2.5-2.6B with reasoning on spent 400 tokens thinking and returned no
  content. `reasoning_effort=none` did nothing. Raising max_tokens to ~1,200
  fixed 2 of 3 cases; the third looped without converging.
- **RWKV-7 G1j** emits its chain of thought as plain text unless run with
  `--reasoning off`.
- **Many small, unrelated requests can make llama-server swap itself to a
  halt.** Each distinct prompt keeps a KV-cache checkpoint (up to 32, ~39 MB
  each). A batch translation job drove generation from ~3 tok/s to 0.06 tok/s
  and put the process in uninterruptible disk wait. `cache_prompt: false`
  didn't stop checkpoints being created; restarting the model did.
- **Coding-agent front ends are unusable here.** One agent tool put ~11,500
  tokens of tool definitions into every system prompt; at ~6 tok/s prompt
  processing that's ~30 minutes before the first reply. A minimal chat client
  (aichat) with no forced tools works fine.
- **`llama-cli` with stdin from `/dev/null`** drops into an endless interactive
  loop even with `--no-conversation`, which looks like a 100x slowdown. Use
  `llama-bench` for speed and `llama-server` for one-shot output checks.
- **`llama-cli` prints tok/s to one decimal.** At ~1.7 tok/s that's a 6% step,
  bigger than many effects being measured.
- **Thermals:** pin the CPU governor to `performance`, log temperatures, and
  run comparisons in both orders, so the "faster" build isn't just the one
  that ran while the chip was cold.
- **bitnet.cpp regression:** a July 2026 submodule update mislabelled i2_s
  tensors at load time and fell off the fast path (0.8/0.7 tok/s instead of
  11.8/2.46). Pinning the previous submodule commit fixed it. The TL1 kernel
  path was slower and crashed on multi-token prompts.
- **Official BitNet-b1.58-2B-4T in mainline llama.cpp** needed a local patch
  (its FFN is `relu(gate)² x up`, not SiLU-gated). With the patch, TQ1_0/TQ2_0
  output was coherent.
- **Bonsai ternary files come in two layouts.** Mainline llama.cpp's Q2_0 is
  group-64 (`*_Q2_0_g64.gguf`); the PrismML fork's older files are group-128
  and fail to load in mainline.

## Tool routing with tiny models

Job: decide whether a spoken question needs a tool (weather, time, date,
dice) or should go to the chat model. **A wrong tool call is worse than a
miss**, because a miss falls through to the chat model, while a wrong call
speaks a confident wrong answer.

| approach | result |
|---|---|
| LFM2.5-230M Q8_0 with tool schemas | reliable up to ~3 tools per request; adding weather to the same request dropped all tools to ~10% hits; weather alone ~70–90% (city-dependent) |
| same, QAD-Q4_0 | +61% generation, but the job is prompt-bound so latency was a wash, and it missed tools the Q8_0 caught |
| Needle 2 (45M) | fast (0.2–0.3 s), but false positives ("capital of France" → weather in "France") and parsed "d20" as a 2-sided die |
| Needle 3 (29–121M) | tool hits 25/25, d20 fixed, but abstained on only 1–4 of 12 non-tool questions |
| **LFM2.5-1.2B, one call, grammar-constrained to `none \| weather \| time \| date \| dice`, "none" listed first** | **16/18** on a wide set, 10/10 end-to-end, ~0.7–4 s |

What made the difference:
- **Small models pick the first option when unsure.** Across 4 orderings x 12
  questions, 11 of 12 wrong answers were exactly the first option listed.
  Putting "none" first makes that bias safe.
- **Telling a tiny model what *not* to do made it worse**: 16/18 → 13/18, and
  "never call this for jokes" made Needle 3 call weather for a joke.
- **Use temperature 0** for any selection task. At default sampling the same
  question hit the tool 2 times in 5.
- **Use code for numbers.** Asked for dice sides, the 1.2B model said 6 for
  "d20" and "twelve sided" alike; a regex got 8/8.
- **Tool description wording is fragile in both directions.** Spelling out
  trigger phrases fixed "what time is it", but a longer weather description
  dropped city hits from 7/7 to 2/7.

## If you have a Pi 5 or newer (expectations, not measured)

The Cortex-A76 has dotprod, so several failed ideas above deserve a retest
there: Q4_0 repacking, `GGML_NATIVE=ON`, and speculative decoding with more
lookahead. Measure RAM bandwidth first (a 20-line streaming-read C program is
enough) and divide by the model size: if llama.cpp is already near that
number, only a smaller model will help.
