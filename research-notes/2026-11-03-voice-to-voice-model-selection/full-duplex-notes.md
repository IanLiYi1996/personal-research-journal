# Full-duplex speech-to-speech sidecar — 2026-11-03

Scope: PersonaPlex-7B-v1 versus Kyutai Moshi; primary official sources only. Qwen/MiniCPM comparisons belong to the main research. No models installed or run. Applicable ancestor AGENTS.md checks found none.

## Recommendation boundary

For **English, native simultaneous listening/speaking with configurable role and voice**, PersonaPlex is the stronger supported choice between these two: NVIDIA adds text role prompts and audio voice conditioning to Moshi. Both support overlapping speech and interruptions. This is a use-case conclusion, not evidence of a universal open-model champion. [NVIDIA project](https://research.nvidia.com/labs/adlr/personaplex/), [PersonaPlex repository](https://github.com/NVIDIA/personaplex), [Moshi paper](https://arxiv.org/abs/2410.00037).

Neither has officially claimed Chinese conversational support: PersonaPlex explicitly specifies English input → English output; Moshi lists English and warns of limited complex-task abilities, no tool access, and a fixed voice per checkpoint. Do not recommend either as the best **Chinese** voice-to-voice model. [PersonaPlex card](https://huggingface.co/nvidia/personaplex-7b-v1), [Moshi card](https://huggingface.co/kyutai/moshiko-pytorch-bf16).

## Verified chronology

- PersonaPlex weights and GitHub: **2026-01-15**, explicitly dated in the card; project announcement same day. Paper submission: **2026-01-14**, per arXiv's displayed submission history. [Card](https://huggingface.co/nvidia/personaplex-7b-v1), [announcement](https://research.nvidia.com/labs/adlr/personaplex/), [paper metadata](https://arxiv.org/abs/2602.06053).
- Moshi prototype/demo unveiled **2024-07-03**; the contemporaneous announcement says code/weights would follow, so this is **not** a verified weight-release date. Its paper's first submission is **2024-09-17**, revised **2024-10-02**. [Kyutai press release](https://kyutai.org/cp_moshi.pdf), [paper metadata](https://arxiv.org/abs/2410.00037).

## Measured performance: separate released weights from experiments

NVIDIA's paper Table 6 / card describe the **released PersonaPlex checkpoint**. Moshi comparator numbers below come from the same paper's Table 2. Values are authors' evaluations, not independently reproduced. TOR = takeover rate; ↓ desirable for pauses/backchannels, ↑ desirable for smooth handover/interruption. [PersonaPlex paper, Tables 2 and 6](https://arxiv.org/html/2602.06053v1).

| Full-Duplex-Bench metric | PersonaPlex released | Moshi |
| --- | ---: | ---: |
| Pause synthetic TOR ↓ | 0.358 | 0.985 |
| Pause Candor TOR ↓ | 0.431 | 0.980 |
| Backchannel TOR ↓ | 0.273 | 1.000 |
| Backchannel frequency ↑ | 0.042 | 0.001 |
| Smooth handover TOR ↑ | 0.908 | 0.941 |
| Smooth handover latency ↓ | 0.170 s | 0.265 s |
| User interruption TOR ↑ | 0.950 | 1.000 |
| Interruption answer GPT-4o score ↑ | 4.290 | 0.765 |
| Interruption stopping latency ↓ | 0.240 s | 0.257 s |

Latency definitions: smooth handover = user stops → agent starts; interruption = user barges in → agent stops. These are **not** first-token/codec latency. Released speaker similarity is 0.650. [Official model card](https://huggingface.co/nvidia/personaplex-7b-v1).

Released-checkpoint human naturalness DMOS: **2.95 ± 0.25 versus Moshi 2.44 ± 0.21** (95% CIs; Table 7). Main experimental checkpoint instead scored **3.90 ± 0.15 versus 3.11 ± 0.15** (Table 1); separate annotator pools make the tables **not directly comparable**. Service-Duplex-Bench **4.48 versus 1.75** is the experimental model's GPT-4o-judged score, **not a separately established released-checkpoint result**. PersonaPlex therefore improves important dimensions, but not every success rate. [Paper, Tables 1, 4, 7 and Appendix A](https://arxiv.org/html/2602.06053v1).

Moshi's original paper reports **160ms theoretical / 200ms practical latency**. Do not directly compare this streaming-model figure against PersonaPlex's end-of-turn 170ms benchmark metric. [Moshi paper abstract](https://arxiv.org/abs/2410.00037).

## Code versus weights; exact hardware wording

| Item | PersonaPlex | Moshi |
| --- | --- | --- |
| Code | MIT | Python/web MIT; Rust Apache |
| Weights | NVIDIA Open Model License; card additionally references CC-BY-4.0 underlying Moshi | CC-BY-4.0 |
| Card use positioning | Ready for commercial use, subject to governing terms | Card describes research use and discourages professional advice/duties; do not conflate this usage guidance with a separate noncommercial weight license |

Sources: [PersonaPlex repo license section](https://github.com/NVIDIA/personaplex#license), [PersonaPlex card governing terms](https://huggingface.co/nvidia/personaplex-7b-v1), [Moshi repo license section](https://github.com/kyutai-labs/moshi#license), [Moshi card](https://huggingface.co/kyutai/moshiko-pytorch-bf16).

- **PersonaPlex:** exact card wording: “Test Hardware: NVIDIA A100 80 GB”. Supported architectures listed are “NVIDIA Ampere (A100)” and “NVIDIA Hopper (H100)”, with Linux/PyTorch. **80GB is tested hardware, not a declared minimum VRAM requirement.** [Card](https://huggingface.co/nvidia/personaplex-7b-v1). Official repo provides `--cpu-offload` with `accelerate` for insufficient GPU memory; it does not give a numeric lower bound or guarantee offloaded real-time performance. [Repo](https://github.com/NVIDIA/personaplex#launch-server).
- **Moshi:** exact README wording: “you will need a GPU with a significant amount of memory (24GB).” This statement concerns PyTorch. MLX was tested on a “MacBook Pro M3”; official MLX q4/q8/bf16 checkpoints exist. README's claim of no PyTorch quantization conflicts with its own experimental q8 listing, so preserve that caveat. Windows lacks official support. [Repo requirements and model list](https://github.com/kyutai-labs/moshi).

## Newer-candidate limit

**Step-Audio-3-Duplex is unverified:** the supplied Step-Audio-2 GitHub URL returned 404 and the exact claimed open-model name/date was not corroborated. Exclude from recommendation; no weights/license/hardware evidence established. StepFun's official site has a differently named StepAudio 3 Realtime product, but that alone does not establish openly released weights and it is outside this open-model comparison. [Supplied repo URL](https://github.com/stepfun-ai/Step-Audio-2), [official StepFun audio site](https://audio.stepfun.ai/).

No claim that PersonaPlex remains the global best open full-duplex model on the cutoff date is justified by this sidecar. Main research should integrate its separately verified Chinese-native newer candidates.
