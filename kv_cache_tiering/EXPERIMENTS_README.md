# vLLM KV Cache Tiering - Active Experiments Suite

This document formalizes the five massive execution arrays currently deployed to the PSC Bridges-2 remote cluster, structured to definitively validate the performance of the Attention-aware Eviction kernel.

---

## 1. Qwen Parameter Scaling Study (Priority 1)

**Hypothesis**: The performance advantage of attention-aware eviction directly scales with foundational model size as layers deepen and discriminator capabilities compress. 
**Hardware**: V100 32GB (Isolated GPU-shared queue)
**Array Specifications**:
- **Qwen 2.5 1.5B** (`scripts/run_qwen_scaling_study.sh` - 4 Hours) → Utilizes *0.15* GPU Memory
- **Qwen 2.5 3B** (`scripts/run_qwen_scaling_study.sh` - 5 Hours) → Utilizes *0.25* GPU Memory
- **Qwen 2.5 7B** (`scripts/run_qwen_scaling_study.sh` - 6 Hours) → Utilizes *0.50* GPU Memory
**Workloads**: ShareGPT, MS-MARCO, HumanEval
**Expected Outcome**: 1.5B provides ~9% improvement; 7B provides >14% improvement.

---

## 2. Long-Context Stress Sweeping (Priority 3)

**Hypothesis**: Under massive multi-document reasoning windows, standard `LRU` blindly destroys dense token semantics, while Attention-aware mathematically preserves exact narrative context structures.
**Hardware**: V100 32GB (Isolated `04:00:00` block)
**Array Specification**:
- **Model**: Qwen 2.5 3B
- **Sequence Sweep**: 4096 (4K) exponentially scaling up to **131,072 (128K) Tokens**.
- **Execution Script**: `scripts/run_long_context.sh`
- **Output**: `benchmark_results/long_context_3b.json`
**Constraints**: Evaluator dynamically overrides the native 32K context cap utilizing `VLLM_ALLOW_LONG_MAX_MODEL_LEN`.

---

## 3. Attention & Score Visual Rendering (Priority 4)

**Hypothesis**: Proving fundamentally exactly *why* Attention-aware policies succeed by visually mapping block scores.
**Hardware**: Executed Extraneously (Local Host via MacOS Python Graphics)
**Execution Strategy**:
- Data extracted via Synthetic pipeline generator (`scripts/collect_eviction_data.py --use-synthetic`)
- Processed cleanly by `matplotlib` & `seaborn` plotting tools into the local `visualizations/` folder.
**Expected Outbound**:
1. `attention_heatmap.png` (Revealing erroneous LRU evictions in Red)
2. `score_distribution.png` (Statistically differentiating kept blocks and discarded blocks: e.g. 0.47 vs 0.03 delta margin).

---

## 4. ROUGE-L Output Quality Validation (Priority 5)

**Hypothesis**: Destructing cache matrices does not probabilistically distort the resulting Language output semantics during long generation workflows.
**Hardware**: V100 32GB (Isolated `02:00:00` block)
**Array Specification**:
- **Model**: Qwen 2.5 3B
- **Verification Engine**: `ROUGE-L` (Lexical Similarity target: >0.98), `BERTScore` (Semantic Similarity target: >0.95), Exact Matching 
- **Execution Script**: `scripts/run_validate_quality.sh`
**Constraints**: Sample temperature dynamically fixed to `0.0` to eliminate generation variance across parallel policy testing boundaries.
