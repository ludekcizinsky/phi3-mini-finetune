# Plan

## TODOs

**Must-Do**:

- [x] Decide which datasets to finetune on (SciQ, EPFL, ...)
- [x] Prepare datasets into required format (CoT + Final Answer)
- [x] Implement fine-tune pipeline (`SFTTrainer`)
- [x] Implement benchmarking suite for MCQ
- [x] Write report
- [X] Quantise model + check that performance doesn't drop too much
- [x] How to extract the answer from the model (in `dpo_model.py`, get some inspiration from common benchmarks)

**Could Do**:

- [x] Check if fine-tuning from base model yields better results (No EPFL DPO)
- [x] In the analysis section of the report, we could look into distribution of the answers
- [ ] Checkout this visualisation [library ](https://github.com/jalammar/ecco) and include some insights into the analysis section
- [ ] Include the DEMO of our final model at HF
- [ ] Take inspiration from this [blog post](https://www.philschmid.de/optimizing-transformers-with-optimum) and benchmark the accuracy and inference speed of the original and quantised model
- [ ] Check out [ORPO](https://huggingface.co/blog/mlabonne/orpo-llama-3)

## Timeline

Sunday, 9th of June:

- [x] Decide on datasets + prepare to expected format for SFTTrainer
- [x] Implement fine-tuning
- [x] Implement benchmarking suite

Monday, 10th of June:

- [x] Implement `mmlu-stem`, `ai2_arc`, `gpqa`, `openbookqa` as datasets
- [x] Implement `mcqa_step` and evaluate on EPFL MCQ examples (maybe also look into making a custom task)

Tuesday, 11th of June:

- [X] Get started on parts of the report that we can already talk about (related work, approach, experiments [data, eval method, baselines, exp details, results], ethical considerations)
- [X] Have the working version of the evaluation pipeline (mcqa_step) ready and make sure it corresponds to the numbers from the official benchmark (baseline (55 to 65) + one of the finetuned models (e.g. sciq 20 to 30))
- [X] Have GPTQ quantisation working for base model
- [X] Have GPTQ quantisation working for finetuned model
- [X] Finetune the default configurations using our HF pipeline on the varoiou datasets and evaluate using our benchmarking suite

Wednesday, 12th of June:

- [X] Quantise the `phi3-arc3` model to 4,3 and 2 bits (naming: `cs552-mlp/phi3-lora-arc3-4bits` etc.)
- [X] Evalute the quantised versions of the `phi3-arc3` model

- Report writing:
    - [X] Introduction (Ludek / Mika)
    - [X] Related work (Ludek)
    - [X] Approach (0. Overall goal of the project 1. determine best base model 2. DPO alignmen 3. SFT)
    - [X] Results

Thursday, 13th of June:

- Finish the first draft of the report
    - [x] Abstract
    - [x] Experiments (Add DPO metrics to eval section, add details about our experimental setup for DPO and SFT, add results from DPO to the results)
    - [x] Analysis (see the bullet points in latex)
    - [x] Ethical Considerations
    - [x] Conclusion

Friday, 14th of June:

- [ ] Polish the README
- [ ] Verify that we can run the `evaluator.py` script for the `mcqa` and `quantisation` tasks
- [ ] Polish the HF organistion files
- [ ] Polish the report