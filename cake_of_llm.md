# Guide: How to conduct systematic experiments with LLMs
This document explains a structured methodology for running fair, reproducible experiments with large language models (LLMs) (including agentic workflows).

*Last update: 02/10/2025*

---

## 1. Research Setup

**Research Question**

> *What prompt and LLM best solves `<problem X>`?*

* If you can break `<problem X>` into smaller steps, evaluate each step in isolation first.

**Hypotheses**

* Structured prompts will outperform minimal prompts.
* Larger LLMs will trade accuracy for latency.

---

## 2. Methodology Overview

There are **3 main stages**:

1. Setting up the environment
2. Running a preliminary analysis to define 3 strong prompts and best LLM candidates
3. Running all experiments (user_prompts) with the best LLM candidates

---

### Stage 1: Setting Up the Environment

1. **Install & Configure Ollama**
   * Download and install Ollama (ollama.com)
   * Write a Python function to call LLMs via Ollama.
   * Make sure the function outputs results in **valid JSON**.

2. **Understand Hardware Constraints**

   * VRAM vs. LLM size vs. context length vs. time vs. accuracy.
   * Example: (i) RTX 4090 with 24GB VRAM, (ii) `llama3.2 3B` (2GB) Q4_K_M (`ollama show <model> --modelfile`) **THEN** `OLLAMA_NUM_PARALLEL=8`, `num_ctx=8192` 
   * Consider **parallel processing** and **batch size**.
  
3. **Fix Decoding Parameters for Reproducibility**
  > ⚠️ Ideally you should use the following paramethers to make the results deterministic.

   - `temperature = 0` → *no creativity, maximum consistency*
   - `top_p = 1` → *include full token distribution*
   - `top_k = 0` → *no cutoff, use all tokens*
   - `seed = 1234` → *same prompt + same seed = same output*
   - `num_ctx = 8192` → *fixed for parallelization*
   - optional:  `keep_alive = 30m` → *avoid frequent reloads.* 
     
   > ⚠️ For brainstorming/pseudocode, higher values like `temperature=0.7, top_p=0.9` may be useful, but results will not be deterministic.
   > ⚠️ Default values for ChatGPT are `temperature=1, top_p=1, top_k= NONE/infinite`
   
---

### Stage 2: Preliminary Analysis

**Goal**: Define and refine **3 strong system prompts** using different levels of detail AND observe the best LLM candidates.

* Prompt styles:
  1. `Task + Format`
  2. `Role + Task + Format`
  3. `Role + Task + Format + Example`

**Steps**

1. Define the **task** clearly, like explaining to a human step-by-step.
2. Collect **5 inputs to test (user prompts)**.
3. Test across **~20 LLMs** of different sizes and families ([Ollama model list](https://ollama.com/search)).

> ⚠️ The expected number of experiments are: 3 [system_prompts] * 5 [user_prompts] * 20 [LLMs] = 300 experiments
> ⚠️ At least one LLM should reach **100%** on your main evaluation metric.

**Evaluation Metrics** (depending on task):

* Accuracy, F1, Precision/Recall, BLEU (translation), ROUGE (summarization).

**Output Requirements**

* JSON must include:
  * `confidence` → `"high"` or `"low"`
  * `reason` → reasoning explained step-by-step (*chain of thought*)

* Function to Validate and fix a couple of cases (JSON):
  * Must not be a string with ```json fences.
  * Predefined keys must exist (ex. `output`, `confidence`, and `reason`).
  * Try up to **3 attempts** to auto-correct mistakes. Otherwise, return an error.

**Data Preprocessing**

* Inputs should not contain unnecessary noise.
* Example: Instead of raw HTML, filter relevant fields and convert to Markdown.
* Ask yourself: *Would a human be able to solve this from the input?*

**Example of expected summary showing main metric and latency [s]:**

| #  | Model                | Prompt 1   | Prompt 2   | Prompt 3   |
|----|----------------------|------------|------------|------------|
| 1  | Gpt-oss_20b          |            |            |            |
| 2  | Mistral-small3.2_24b |            |            |            |
| 3  | Qwen3_4b             |            |            |            |
| 4  | Qwen3_8b             |            |            |            |
| 5  | Qwen3_14b            |            |            |            |
| 6  | Llama4_16x17b        |            |            |            |
| 7  | Llama4_128x17b       |            |            |            |
| 8  | Gemma3_4b            | 60% \| 0.4 | 80% \| 0.4 | 100% \| 0.4|
| 9  | Gemma3_12b           | 80% \| 0.8 | 100% \| 0.8| 100% \| 0.8|
| 10 | Deepseek-r1_8b       |            |            |            |
| 11 | Deepseek-r1_14b      |            |            |            |
| 12 | Phi4_14b             |            |            |            |
| 13 | Llama3.2_1b          |            |            |            |
| 14 | Llama3.2_3b          |            |            |            |
| 15 | Llama3.1_8b          |            |            |            |
| 16 | Mistral_7b           |            |            |            |
| 17 | Qwen2.5_1.5b         |            |            |            |
| 18 | Qwen2.5_3b           |            |            |            |
| 19 | Qwen2.5_7b           |            |            |            |



**Aspects to discuss**
a. What are the best results?
b. Check relation between model size and latency.
c. Measure power usage (if possible).
d. Measure memory usage (VRAM footprint) (`nvidia-smi peak`).

---

### Stage 3: Selecting the Best LLMs

1. **Choose top models from previous stage**
2. **Run on the entire dataset**.
3. **Analyze results**
> ⚠️ If 'best model' isn’t perfect, **manually review mistakes**. Can better prompting fix them? Try improving prompts.

b. Compare error intersection of errors:

   |      | LLM1 | LLM2 | LLM3 | LLM4 |
   | ---- | ---- | ---- | ---- | ---- |
   | LLM1 | 0    | 0    | 0    | 0    |
   | LLM2 | 0    | 10   | 9    | 10   |
   | LLM3 | 0    | 9    | 20   | 7    |
   | LLM4 | 0    | 10   | 7    | 30   |

c. Investigate error related to input length (bin prompts by size, check error rates).

d. Investigate whether `confidence=high` correlates with correctness.

e. Investigate error categories: Formatting error, Hallucination, Reasoning flaw

f. Other visualizations: Bar plots (accuracy by model), Violin plots (latency distributions), Heatmaps (error overlaps, time vs accuracy trade-offs)

