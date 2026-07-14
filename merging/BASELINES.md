# Reproducing the Figure 5 comparison baselines

Figure 5's accuracy/memory bars compare **SANDHI** against three baselines the
SANDHI *merge* pipeline does not itself emit. Each is reproducible; the numbers
that ship in `plots/data/` are these baselines pre-computed. All three evaluate
each model on its registry benchmark (see `plots/README.md` / `FIGURE_MODEL_MAP`).

## 1. No-merge (base model) — the "before" reference
Evaluate the unmerged model on its domain benchmark.
```bash
lm_eval --model hf --model_args pretrained=<source-or-base-model> \
        --tasks <benchmark> --batch_size auto
```
Ships in `plots/data/vllm-no-merge.csv`. (≈ the pipeline's point-A / per-model
MICR baseline; also the "Base" column in the LoRA guide below.)

## 2. Full-merge = Multi-SLERP [24] — merges ALL layers (no selective strategy)
The paper's full-merge baseline is **Multi-SLERP** (ref [24]): a single model
formed by spherically interpolating **all** of a pool's source models across
**every** layer — the opposite of SANDHI's *selective* per-slot merging. Compute
with [mergekit](https://github.com/arcee-ai/mergekit)'s `multislerp` method —
**not** in the reference image, so `pip install mergekit` first (vLLM + lm_eval
ARE in the image):
```bash
# merge.yml — Multi-SLERP of the pool's fine-tunes onto their shared base:
#   merge_method: multislerp
#   base_model: <shared base, e.g. meta-llama/Llama-3.1-8B>
#   models: [<fine-tune 1>, <fine-tune 2>, ...]   # the pool's source models (all layers)
mergekit-yaml merge.yml ./fullmerge_out --cuda
# then eval the merged model on each pool benchmark:
lm_eval --model hf --model_args pretrained=./fullmerge_out --tasks <benchmark>
```
Ships in `plots/data/full_merge/{llama3.1,qwen2_5,qwen3}_domain_results.csv`
(the plot consumes the `multi_slerp` column).

## 3. LoRA (rank-128 adapters served on the base via vLLM)
Adapters live at **`anjohn0077/NEXS-lora-adapters`** — a manifest repo pointing
to per-domain adapters (`anjohn0077/NEXS-<domain>-lora`, + six Qwen3-32B ones).
`manifest.json` maps domain → source model → adapter repo → benchmark, and that
repo's **README is the full serving/eval guide**. Essentials (Llama family):
```bash
# fetch adapters:
for d in finance legal medical toxicity truthfulness; do
  hf download "anjohn0077/NEXS-${d}-lora" --local-dir "./adapters/${d}"; done

# serve base + all adapters on one vLLM server:
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B \
  --enable-lora --max-loras 5 --max-lora-rank 128 --gpu-memory-utilization 0.85 --port 8000 \
  --lora-modules finance=./adapters/finance legal=./adapters/legal \
                 medical=./adapters/medical toxicity=./adapters/toxicity \
                 truthfulness=./adapters/truthfulness

# eval each adapter against the running server:
lm_eval --model local-completions --tasks mmlu_econometrics --output_path results/vllm_finance \
  --model_args model=finance,base_url=http://localhost:8000/v1/completions,tokenizer=meta-llama/Llama-3.1-8B,num_concurrent=10
#   legal→mmlu_professional_law · medical→medqa_4options
#   toxicity→sst2 · truthfulness→truthfulqa_mc2
```
The **Qwen3-32B family** is analogous (`--model Qwen/Qwen3-32B`, six adapters,
`tokenizer=Qwen/Qwen3-32B`; IF→ifeval, medical→medqa_4options, russian→m_mmlu_ru).
Ships in `plots/data/lora/{llama_adapter_acc,qwen32b_adapter_acc}.csv`; the repo
README also carries the full Base / LoRA / full-fine-tune result tables.

## Notes
- **`mergekit`** is the only extra dependency (needed for §2; `pip install mergekit`).
  vLLM and lm_eval (§1, §3) are already in the reference image.
- Base model `meta-llama/Llama-3.1-8B` is **gated** — `hf auth login` / `HF_TOKEN` first.
- These feed the three non-SANDHI bars in the Fig 5 plots (`plots/README.md`); the
  SANDHI bars come from the pipeline's `report.csv`.
