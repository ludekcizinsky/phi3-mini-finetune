## LoRA

Finetuning full LLM is costly (memory, time). Motivated by this issue, LoRA proposes a new method where small adaptive layers are added to the existing pretrained base LLM. Thus, instead of finetuning the full LLM, only the adaptive layers are trained. This method is shown to be more efficient than finetuning the full LLM, while achieving similar performance. You can read more about the method in our [summary](lora_summary.pdf) of the original paper.

