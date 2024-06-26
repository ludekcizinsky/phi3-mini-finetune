## V1 - V3

In these phases, we were getting familiar with the codebase and the data, figuring out optimal setup.

## V4

At this point, we have a reliable codebase and know how much we can afford to train in a reasonable time. For this reason, as our defaults unless otherwise specified, we will use the following configuration:

```
wandb.enabled=True wandb.group="section name" trainer.args.max_steps=250 trainer.args.logging_steps=5 trainer.args.learning_rate=4.0e-05 trainer.beta=0.4 trainer.loss_type="sigmoid" trainer.label_smoothing=0.1 peft_config.r=32 recipe.max_eval_samples=640 recipe.train_datasets.0.filtering_strategy=none
```

We found this learning rate as a good starting based on the online discussions. Further, the given batch size allows us to utilise the memory of the GPU to the maximum without crashes. Finally, we deciced to use 250 steps which means we go over 4000 samples, which is roughly 20 % of the training data. We start with the rank 32 as the default value in lora which gives us a reasonable size of the model that needs to be adapted. Finally, we set the DPO beta to be 0.4. The range of beta is usually between 0 and 0.5 where 0 means we make the policy model less sensitive to the difference to the underlying reference model while higher beta means we do not want to diverge too much from the reference model. For this reason, we decided to take a conservative approach and set the beta to 0.4 thinking this is a good starting point. Importantly, we will focus for now solely on optimising performance for the phi3 model.

In the remainder of this section, we will run different ablation studies and observe how they affect the evaluation rewards accuracy which is they key metric (note to us: this is what is used by the mnlp team as the final metric).

### V4.1 - Varying data filtering methods

We suspect some student annotations to be noisy, i.e. they are not sure of the correct answer or one response is clearly worse than the other. Further, for each question, we have multiple annotations. We have implemented various filtering strategies for the preference data. In these strategies, we often refer to a `score`. The score for given sample is computed as the agreement between `overall` (which can be either only one of the samples) and 4 other ranking subcategories that include `clarity`, `correctness`, `relevance` and `completness`. Thus, the maximum score is 4 and the minimum is 0. The filtering strategies are:

- `none`: No filtering is applied.
- `keep_first`: Only the first preference pair for each question is kept.
- `global_threshold`: this setup has a special parameter `mode`:
  - `least`: Only the preference pairs with a score **at least** the `threshold` are kept.
  - `most`: Only the preference pairs with a score **at most** the `threshold` are kept.
- `local_tolerance`:
  - `least`: the function will keep all rows where the agreement is **at least** the maximum agreement found for a question in the group **minus** the `tolerance`
  - `most`: the function will keep all rows where the agreement is **at most** the maximum agreement found for a question in the group **plus** the `tolerance`

We will use the following training hyper-parameters:

1. `none` - baseline
2. `keep_first`
3. `global_threshold` (`mode=least` and `threshold=4`)
4. `global_threshold` (`mode=least` and `threshold=3`)
5. `local_tolerance` (`mode=least` and `tolerance=0`)
6. `local_tolerance` (`mode=least` and `tolerance=1`)

**Find the results [here](https://wandb.ai/ludekcizinsky/mnlp-project/table?nw=e0hf2n9ovd)**.

### V4.2 - Varying the dpo loss

Our next ablation will be to vary the `DPO loss` function (see more [here](https://huggingface.co/docs/trl/main/en/dpo_trainer#loss-functions)), we will use the same training hyper-parameters as in the previous experiment:

`lr=4.0e-5, b=16, g=1, s=250, r=32`

and then vary the loss functions as follows:

1. `loss_type=hinge beta=0.05` (margin = 1/beta)
2. `loss_type=hinge beta=0.1`
3. `loss_type=ipo beta=0.05`
4. `loss_type=ipo beta=0.1`

**Find the results [here](https://wandb.ai/ludekcizinsky/mnlp-project/table?nw=zh5fpm5zenk)**.

### V4.3 - Label smoothing

We will now try to apply label smoothing to the model, i.e., we assume that preference labels are noisy with some probability that we pass as the parameter. We will use the following training hyper-parameters:

`lr=4.0e-5, b=16, g=1, s=250, r=32 beta=0.4`

and then vary the label smoothing parameter:

1. `label_smoothing=0.0`
2. `label_smoothing=0.1`
3. `label_smoothing=0.2`
4. `label_smoothing=0.3`
5. `label_smoothing=0.4`

**Find the results [here](https://wandb.ai/ludekcizinsky/mnlp-project/table?nw=jhrqwpuib5)**.

### V4.4 - LR vs LoRA rank

In this section, we aim to understand the role of size of the model defined by the lora rank and the learning rate. We will try to vary the learning rate and the rank of the lora model. We will use the following configurations:

**r=16**:

1. `lr=1.0e-5, s=250, r=16`
2. `lr=2.0e-5, s=250, r=16`
3. `lr=4.0e-5, s=250, r=16`
4. `lr=8.0e-5, s=250, r=16`

**r=32**:

1. `lr=1.0e-5, s=250, r=32`
2. `lr=2.0e-5, s=250, r=32` ludek (scheduled)
3. `lr=4.0e-5, s=250, r=32` pierre (scheduled)
4. `lr=8.0e-5, s=250, r=32` pierre (scheduled)

**r=64**:

1. `lr=1.0e-5, s=250, r=64` pierre (scheduled)
2. `lr=2.0e-5, s=250, r=64` pierre (scheduled)
3. `lr=4.0e-5, s=250, r=64` pierre (scheduled)
4. `lr=8.0e-5, s=250, r=64`

**r=128**:

1. `lr=1.0e-5, s=250, r=128`
2. `lr=2.0e-5, s=250, r=128`
3. `lr=4.0e-5, s=250, r=128`
4. `lr=8.0e-5, s=250, r=128`

**Find the results [here](https://wandb.ai/ludekcizinsky/mnlp-project/workspace?nw=7x97vpp5cbx)**.

### V4.5 - Varying dpo beta

Let's now observe how the model's performance changes if we ablate the `dpo beta` parameter. We will use the following values:

1. `beta=0.05` sc
2. `beta=0.1` sc
3. `beta=0.15` sc
4. `beta=0.2` sc
5. `beta=0.25` rai
6. `beta=0.3` rai
7. `beta=0.35` rai
8. `beta=0.4` rai
9. `beta=0.45` rai
10. `beta=0.5` rai

**Find the results [here](https://wandb.ai/ludekcizinsky/mnlp-project/workspace?nw=era8ketc6p)**.

## V5

The goal of this phase is to find the best model for submission.

### V5.1 - Full Runs

In this phase, we will use the best peforming parameters from the previous experiments and run them on the full training data. In addition, we will enable evaluation during training on subset of validation dataset, and then we will evaluate the model on the full validation dataset at the end. We will use the following configurations:

| Model | LR   | Rank | Loss | Beta | LB Smoothing | Data Filt. | Val. Accuracy (%) |
| ----- | ---- | ---- | ---- | ---- | ------------ | ---------- | ----------------- |
| H4    | 4e-5 | 32   | IPO  | 0.1  | 0.1          | None       | **67.01%**        |
| M2    | 2e-5 | 16   | IPO  | 0.1  | 0.0          | None       | 66.61%            |
| H2    | 4e-5 | 32   | DPO  | 0.05 | 0.1          | None       | 65.97%            |
| H1    | 2e-5 | 16   | DPO  | 0.4  | 0.1          | None       | 64.76%            |
| M1    | 2e-5 | 16   | IPO  | 0.1  | 0.0          | LT (λ=0)   | 64.69%            |
| H3    | 4e-5 | 32   | DPO  | 0.4  | 0.0          | None       | 63.89%            |
| H5    | 4e-5 | 32   | DPO  | 0.4  | 0.1          | LT (λ=0)   | 62.07%            |

**Find the detailed results [here](https://wandb.ai/ludekcizinsky/mnlp-project/workspace?nw=t1fz1hjq34d)**.

### V5.2 - Final Run

Given the previous phase, we found the best perfmring model to be `H4`, therefore we will now run this model on a full dataset, i.e., combining training and validation into one dataset.

**Find the results [here](https://wandb.ai/ludekcizinsky/mnlp-project/workspace?nw=rssf7mrdnp)**.

