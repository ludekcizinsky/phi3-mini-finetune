## 🤏 Size Does Not Matter: A Data-Centric Approach to Fine-Tuning

This project has two main goals. First, we aim to align the open-source model [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) to our collected preference data using Direct Preference Optimization (DPO). Second, we plan to further finetune the model on Multiple Choice Questions (MCQ) dataset to improve the performance of the model on the MCQ task, and then quantise it using [GPTQ](https://arxiv.org/abs/2210.17323) so the model can be deployed even on the phones. Ultimately, we hope that our work will lead to an aligned model capable of assisting EPFL students during their studies.

## 🔗 Shortcuts
Here is a list of things that you likely want to do:
- 🏆 [Evaluate](model/) our model against your own data
- 📚 [Read](pdfs/MLP.pdf) our final report

## 💫 Vibe Check (GPU required)

Before you dive into the nitty gritty details and all those numbers, you may want to check the *vibe* of our finetuned models. First, you have to setup your environment by installing the required [dependencies](model/requirements.txt). Then it is as simple as running the following command:

```bash
cd model
python scripts/chat.py cs552-mlp/phi3-lora-arc3
```

where as the argument you can use any of the models from our [🤗 collection](https://huggingface.co/collections/cs552-mlp/submitted-models-6655ba2295bc4a27ba9b8de5) (see more details below). This will open a simple CLI chat interface where you can interact with the model of your choice.

*Tested on NVIDIA V100 GPU*

## 🤗 Collections
Check our HF page, where we have collected:
- [🔗](https://huggingface.co/collections/cs552-mlp/m2-dpo-aligned-models-6655b7c830511ba6251d65b9) DPO aligned models. Our [progress report](pdfs/dpo_alignment.pdf) talks in detail about the process.
- [🔗](https://huggingface.co/collections/cs552-mlp/m3-sft-for-mcqa-666c47433b570f44d7d91751) MCQA finetuned models
- [🔗](https://huggingface.co/collections/cs552-mlp/m3-quantisation-666c4ac265901a559ff71534) Quantised versions of the best performing MCQ finetuned model
- [🔗](https://huggingface.co/collections/cs552-mlp/submitted-models-6655ba2295bc4a27ba9b8de5) Final submitted models

## 🌝 Results TLDR

We know the feeling, you just want to know the most important details without us waffling around too much. Here is a quick summary of our results:

### Baseline Model Comparison

We first verified that indeed phi3 is the biggest baller among the small open-source models. We evaluated its performance on several standard MCQ datasets including MMLU against Llama3-8b and OpenElm by Apple. The below table shows the accuracy and SE. The results are in line with the public benchmarks showing large gap between OpenELM and the other two models. And Phi3 with Llamma3-8b being very close in performance.

|       | LLama         | OpenELM       | Phi3          |
|-------|---------------|---------------|---------------|
| ARC   | 67.4 ± 1.1    | 50.2 ± 1.2    | **69.4 ± 1.1** |
| GPQA  | 29.8 ± 2.5    | 23.6 ± 2.3    | **33.6 ± 2.5** |
| MMLU  | 54.5 ± 4.0    | 24.0 ± 3.5    | **57.3 ± 3.9** |
| OBQA  | 34.0 ± 2.1    | 25.4 ± 1.9    | **38.8 ± 2.2** |
| SciQ  | **96.3 ± 0.6** | 89.7 ± 1.0    | 95.5 ± 0.7    |

### DPO Alignment
The table below summarizes the best peforming configurations we found during our initial hyper-parameter search on subset of data. We then run these best configurations on the full training dataset and evaluated them on the full validation dataset. Each of the ablation is described in detail in the [report](pdfs/dpo_alignment.pdf).

| Model | LR     | Rank | Loss | Beta | LB Smoothing | Data Filt. | Val. Accuracy (%) |
|-------|--------|------|------|------|--------------|------------|-------------------|
| H4    | 4e-5   | 32   | IPO  | 0.1  | 0.1          | None       | **67.01%**        |
| M2    | 2e-5   | 16   | IPO  | 0.1  | 0.0          | None       | 66.61%            |
| H2    | 4e-5   | 32   | DPO  | 0.05 | 0.1          | None       | 65.97%            |
| H1    | 2e-5   | 16   | DPO  | 0.4  | 0.1          | None       | 64.76%            |
| M1    | 2e-5   | 16   | IPO  | 0.1  | 0.0          | LT (λ=0)   | 64.69%            |
| H3    | 4e-5   | 32   | DPO  | 0.4  | 0.0          | None       | 63.89%            |
| Base  | 4e-5   | 32   | DPO  | 0.4  | 0.1          | None       | 63.27%            |
| H5    | 4e-5   | 32   | DPO  | 0.4  | 0.1          | LT (λ=0)   | 62.07%            |


You can find detailed summary of training and evaluation through [W&B](https://wandb.ai/ludekcizinsky/mnlp-project?nw=t1fz1hjq34d). The corresponding model checkpoints can be found in our [🤗 collection](https://huggingface.co/collections/cs552-mlp/hyperparameter-tuned-models-6655b7c830511ba6251d65b9).


### MCQA Finetuning

We found out that finetuning the DPO aligned model to MCQA does not yield as good performances as if we finetune the Phi3 model directly. The table below shows again the accuracy and SE on various MCQ datasets. Each columns includes a model trained on a different dataset. As it is clear from the table, the most "Ws" were scored by the Phi3-Arc model. We therefore decided to quantise this model. 


|       | Phi3          | Phi3-Arc      | Phi3-DPO      | Phi3-MCQ      | Phi3-OBQA     | Phi3-SciQ     |
|-------|---------------|---------------|---------------|---------------|---------------|---------------|
| ARC   | 69.4 ± 1.1    | 66.3 ± 1.1    | 66.0 ± 1.1    | 65.9 ± 1.1    | 61.8 ± 1.2    | **71.6 ± 1.1** |
| GPQA  | **33.6 ± 2.5** | 33.2 ± 2.5    | 33.2 ± 2.5    | 33.1 ± 2.5    | 31.8 ± 2.5    | 26.5 ± 2.4    |
| MMLU  | 57.3 ± 3.9    | **57.8 ± 3.9** | 57.0 ± 3.9    | 55.8 ± 3.9    | 55.7 ± 4.0    | 22.1 ± 3.5    |
| OBQA  | 38.8 ± 2.2    | **40.4 ± 2.2** | 37.6 ± 2.2    | 38.2 ± 2.2    | 38.0 ± 2.2    | 37.2 ± 2.2    |
| SciQ  | 95.5 ± 0.7    | 95.4 ± 0.7    | 95.1 ± 0.7    | 97.0 ± 0.5    | 94.1 ± 0.7    | **97.4 ± 0.5** |


### Quantisation

Finally, we got to the quantisation part wondering if [GPTQ](https://arxiv.org/abs/2210.17323) is as much of a baller method as [Elias](https://efrantar.github.io/) and his colleagues claim. Turns out, Elias was cooking differently (probably the reason why he landed the internship at Deepmind...)! As is clear, up to the 4bits, we were able to retain quite a large portion of the original performance. Too bad quantisation will no longer be needed, as we will soon [apparently have](https://arxiv.org/abs/2406.02528) MatMul free transformers 😢.

|       | GPTQ-2b       | GPTQ-3b       | GPTQ-4b       | GPTQ-8b       | Phi3-Arc      |
|-------|---------------|---------------|---------------|---------------|---------------|
| ARC   | 23.8 ± 1.1    | 55.9 ± 1.2    | 64.1 ± 1.2    | **66.4 ± 1.1** | 66.3 ± 1.1    |
| GPQA  | 23.2 ± 2.3    | 28.7 ± 2.4    | 32.0 ± 2.5    | 32.7 ± 2.5    | **33.2 ± 2.5** |
| MMLU  | 24.2 ± 3.6    | 46.0 ± 4.1    | 56.0 ± 4.0    | **57.8 ± 3.9** | 57.8 ± 3.9    |
| OBQA  | 13.0 ± 1.5    | 33.8 ± 2.1    | 37.6 ± 2.2    | 40.0 ± 2.2    | **40.4 ± 2.2** |
| SciQ  | 22.9 ± 1.3    | 93.2 ± 0.8    | 95.5 ± 0.7    | 95.4 ± 0.7    | **95.4 ± 0.7** |

## 🗃️ Docs
This project taught us many valuable things, we tried to document them in the following:
- [🛠️](docs/commands) Useful commands describes range of topics from how to operate on the SLURM cluster to how to push model to HF hub
- [🔬](docs/methods/) Methods document our approach to training (LoRA, Quantisation) as well as benchmarking

## 👨‍👦‍👦 MLP?

Initially, it stood for [Mika](https://www.mikasenghaas.de/)-[Ludek](https://cizinsky.cc/)-Peter, but Peter went to Munich, so now it stands for [Mika](https://www.mikasenghaas.de/)-[Ludek](https://cizinsky.cc/)-Pierre. 