# Benchmarking

## Final Evaluation

For the final evaluation we are going to be evaluated on secret, scientific multiple-choice questions formatted as shown in the official [project code repository](https://github.com/CS-552/project-code-2024/blob/main/datasets/mcqa_example.jsonl). To run, our model has to *output a single letter response*.

Here is an example question:

```
Question:
Statement 1| Linear regression estimator has the smallest variance among all unbiased estimators.
Statement 2| The coefficients α assigned to the classifiers assembled by AdaBoost are always non-negative.

Options:
A. True, True
B. False, False
C. True, False
D. False, True

Answer:
```

We want to both create a *training* dataset and find *evaluation* benchmarks that closely approximate this style of questions.

## Language Model Evaluation Harness

We use the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file) implementation for benchmarking LLMs.

**Reasons**

* Implements >60 academic benchmarks for LLMs in ready-to-use CLI or Python library

+ Support for evaluating models from `transformers` (including evaluating on LoRA adapter + GPTQ quantisation, i.e. all the models that we plan to use), both locally and from the HF Hub
+ Extensible for custom tasks, if necessary
+ Used for the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) by HF

**Basic Usage**

```bash
lm-eval \
    --model hf \
    --model_args pretrained=<model-name> \
    --tasks task1,task2 \
    --device cuda:0 \
    --batch_size auto \
    --output_path <path>
```

This will automatically download the model, evaluate the specified benchmarks sequentially and write the results to the output path. By default, it runs only on the test split of the benchmarks. For exact usage, check out the `model/eval.sh`.

**How are benchmark accuracies computed?**

We only use benchmarks of output type `multiple_choice` which use "loglikelihood-based comparative scoring" of the answer options. Given some prompt (question) and generation (answer options), the library computes the sum of log-probabilities of all continuations (this is equivalent to the probability the language model assigns to each continuation) and uses the highest-scoring continuation as its predictions. There are two ways of computing these log-likelihood continuations:

1. Simple sum of log-probabilities: $\sum_{j=m}^{n_i-1}\log P(x_j |x_{0:j})$
2. Byte-length normalised sum of log-probabilities: $\sum_{j=m}^{n_i-1}\log P(x_j |x_{0:j})/\sum_{j=m}^{n_i-1} L_{x_j}$, where $L_x$ is the number of bytes represented by token $x_j$

The former option is used for returning `acc` and the latter for `acc_norm` for benchmarks of type `multiple_choice`.

## Benchmarks

This is the full list of commonly used LLM benchmarks compiled from the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) (OLL )by HF, Phi-3 Technical Report and the [LLama3-Technical Report](https://ai.meta.com/blog/meta-llama-3/), which are **also available on LMEH**. We give a small description, the number of samples per split, where the benchmark is used and whether or not it is relevant for us to include in the

| Benchmark                     | Description                                                                                                                                                                                                                      | Samples           | Links                                                                                                                                                                                                            | Used In              | Include?        |
| :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | --------------- |
| AI2 Reasoning Challenge (ARC) | A set of grade-school science questions.                                                                                                                                                                                         | 1.12K, 299, 1.17K | [Paper](https://arxiv.org/abs/1803.05457), [HF](https://huggingface.co/datasets/allenai/ai2_arc), [LMEH]()                                                                                                                | OLL, Phi-3           | ✅              |
| HellaSwag                     | Commonsense inference, which is easy for humans (~95%) but challenging for SOTA models                                                                                                                                           | 39.9K, 10K, 10K   | [Paper](https://arxiv.org/abs/1905.07830), [HF](https://huggingface.co/datasets/Rowan/hellaswag), [LMEH]()                                                                                                                | OLL, Phi-3           | ❌ (Off Task)   |
| MMLU                          | Test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more.                                                                       | 99.8K, 1.53K, 14K | [Paper](https://arxiv.org/abs/2009.03300), [HF](https://huggingface.co/datasets/cais/mmlu), [LMEH]()                                                                                                                      | OLL, Phi-3, Llama 3  | ✅              |
| TruthfulQA                    | Test to measure a model's propensity to reproduce falsehoods commonly found online. Note: TruthfulQA is technically a 6-shot task in the Harness because each example is prepended with 6 Q/A pairs, even in the 0-shot setting. | 817               | [Paper](https://arxiv.org/abs/2109.07958), [HF](https://huggingface.co/datasets/truthfulqa/truthful_qa), [LMEH]()                                                                                                         | OLL, Phi-3           | ❌ (Off-Task)   |
| Winogrande                    | Adversarial and difficult Winograd benchmark at scale, for commonsense reasoning                                                                                                                                                 | 44k               | [Paper](https://arxiv.org/abs/1907.10641), [HF](https://huggingface.co/datasets/allenai/winogrande), [LMEH]()                                                                                                             | OLL, Phi-3           | ❌ (Off-Task)  |
| GSM8k                         | Diverse grade school math word problems to measure a model's ability to solve multi-step mathematical reasoning problems                                                                                                         | 7.47K, 1.32K      | [Paper](https://arxiv.org/abs/2110.14168), [HF](https://huggingface.co/datasets/openai/gsm8k), [LMEH]()                                                                                                                   | OLL, Phi-3, Llama 3 | ❌ (No MCQ)     |
| ANLI                          | Large-scale NLI benchmark dataset by Facebook (Predicts Entailment, Neutral, Contradiction)                                                                                                                                      |                   | [Paper](https://aclanthology.org/2020.acl-main.441/), [HF](https://huggingface.co/datasets/facebook/anli), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/anli/README.md)            | Phi-3                | ❌ (Off-Task)  |
| MedQA                         | Multiple choice question answering based on the United States Medical License Exams.                                                                                                                                             | 10.2K, 1.27K      | [Paper](https://arxiv.org/abs/2009.13081), [HF](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/medqa/medqa.yaml)       | Phi-3                | ❌ (Off-Topic) |
| AGIEval                       | Tasks involving historical data or questions related to history and historical texts.                                                                                                                                            |                   | [Paper](https://arxiv.org/abs/2304.06364.pdf), [HF](https://huggingface.co/hails), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/agieval/README.md)                                  | Phi-3                | ❌ (Off-Topic) |
| TriviaQA                      | A large-scale dataset for trivia question answering to test general knowledge.                                                                                                                                                   |                   | [Paper](https://arxiv.org/abs/1705.03551), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/triviaqa/README.md)                                                                      | Phi-3                | ❌ (Off-Topic) |
| PiQA                          | Physical Interaction Question Answering tasks to test physical commonsense reasoning.                                                                                                                                            |                   | [Paper](https://arxiv.org/abs/1911.11641), [HF](https://huggingface.co/datasets/ybisk/piqa), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/piqa/README.md)                           | Phi-3                | ❌ (Off-Topic)  |
| SciQ                          | Science Question Answering tasks to assess understanding of scientific concepts.                                                                                                                                                 |                   | [Paper](https://aclanthology.org/W17-4413.pdf), [HF](https://huggingface.co/datasets/allenai/sciq), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/sciq/README.md)                    | Phi-3                | ❌ (Training)   |
| BigBench                      | Broad tasks from the BIG-bench benchmark designed to push the boundaries of large models.                                                                                                                                        |                   | [Paper](https://arxiv.org/abs/2206.04615), [HF](https://huggingface.co/datasets/google/bigbench), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/bigbench/README.md)                  | Phi-3                | ❌ (Too broad)  |
| OpenBookQA                    | Open-book question answering tasks that require external knowledge and reasoning.                                                                                                                                                |                   | [Paper](https://arxiv.org/abs/1809.02789), [HF](https://huggingface.co/datasets/allenai/openbookqa), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/openbookqa/README.md)             | Phi-3                | ✅              |
| BoolQ                         | N/A                                                                                                                                                                                                                              | N/A               | N/A                                                                                                                                                                                                              | Phi-3                | ❌ (N/A)        |
| CommonSenseQA                 | N/A                                                                                                                                                                                                                              | N/A               | N/A                                                                                                                                                                                                              | Phi-3                | ❌ (N/A)       |
| HumanEval                     | N/A                                                                                                                                                                                                                              | N/A               | N/A                                                                                                                                                                                                              | Phi-3, Llama 3      | ❌ (N/A)       |
| MBPP                          | N/A                                                                                                                                                                                                                              | N/AN/A            | N/A                                                                                                                                                                                                              | Phi-3                | ❌ (N/A)        |
| GPQA                          | GPQA is a multiple-choice, Q&A dataset of very hard questions written and validated by experts in biology, physics, and chemistry.                                                                                               |                   | [Paper](https://arxiv.org/abs/2311.12022), [HF](https://huggingface.co/datasets/Idavidrein/gpqa), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gpqa/README.md)                      | Phi-3, Llama 3      | ✅              |
| Math                          | The MATH dataset consists of problems from mathematics competitions, including the AMC 10, AMC 12, AIME, and more                                                                                                               |                   | [Paper](https://arxiv.org/abs/2103.03874), [HF](https://huggingface.co/datasets/hendrycks/competition_math), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/README.md) | Phi-3, Llama 3       | ❌ (No MCQ))    |

Given these datasets, we select the following datasets for our evaluation based on their closeness to the final evaluation task.

| Benchmark                     | Description                                                                                                                                                | Total Samples | Samples/ Split                                     | Links                                                                                                                                                                                                | LMEH                                                              |
| :---------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| AI2 Reasoning Challenge (ARC) | A set of grade-school science questions.                                                                                                                   | 3.55K         | 2.38K (Arc Easy), 1.17K (Arc Challenge)            | [Paper](https://arxiv.org/abs/1803.05457), [HF](https://huggingface.co/datasets/allenai/ai2_arc), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/arc/README.md)           | ai2_arc                                                           |
| MMLU                          | Test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more. | 3.15K         | *Many splits* splits (Statistics, Physics, ...)) | [Paper](https://arxiv.org/abs/2009.03300), [HF](https://huggingface.co/datasets/cais/mmlu), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu)                          | mmlu_stem                                                         |
| GPQA                          | GPQA is a multiple-choice, Q&A dataset of very hard questions written and validated by experts in biology, physics, and chemistry.                         | 1.19K         | 448 (Main), 546 (Extended), 198 (Diamond)         | [Paper](https://arxiv.org/abs/2311.12022), [HF](https://huggingface.co/datasets/Idavidrein/gpqa), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gpqa/README.md)          | gpqa_diamond_zeroshot, gpqa_extended_zeroshot, gpqa_main_zeroshot |
| OpenBookQA                    | Open-book question answering tasks that require external knowledge and reasoning.                                                                          | 500           | *No splits*                                      | [Paper](https://arxiv.org/abs/1809.02789), [HF](https://huggingface.co/datasets/allenai/openbookqa), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/openbookqa/README.md) | openbookqa                                                        |
| SciQ                    | Scientific QA                                                                          | 1000           | *No splits*                                      | [Paper](https://aclanthology.org/W17-4413.pdf), [HF](https://huggingface.co/datasets/allenai/sciq), [LMEH](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/sciq/README.md) | sciq                                                        |

Hence, the total number of eval samples is roughly **9.39K samples**. Most samples are MCQs with four answer options. Hence, to evaluate the LM we need to extract the log-probs of $9.39\text{ K}\times 4 = 37.56 \text{ K}$ samples. LMHE calls this `loglikelihood requests`. Running the full evaluation suite, takes about **4 hours** on  a Tesla V100 with 32GB of data.

> Note: The GPQA dataset is a gated dataset. To use it, request access [here](https://huggingface.co/datasets/Idavidrein/gpqa) and make sure to log in with a HF Token with write access to this repository.

Could consider: `piqa`

## Issues with LLM Benchmarking

- **Data Leakage**: Models might be trained on benchmarking questions, and as such know the answer by overfitting, instead of solving the question
- **Discrepancy to In-Field Performance**: The benchmarks might not be reflective of real-world use cases
- **Generality vs. Specialisation**: Most benchmarks require rather broad, general knowledge, and as such are not useful in determining the performance in specialised domains

## Questions

- How can we run with 4-bit precision for reference model?
- Can we specify the splits that we run the evaluation on?
