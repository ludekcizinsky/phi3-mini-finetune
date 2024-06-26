## 📤 Upload the finetuned model to the 🤗 Hub

Start with installing the 🤗 Hub CLI:

```bash
pip install huggingface-hub
```

Then, log in to your account:

```bash
huggingface-cli login
```

Finally, push your model to the Hub:

```bash
huggingface-cli upload --repo-type model <repo_id> <local_path>
```

For example:

```bash
huggingface-cli upload --repo-type model cs552-mlp/phi3-dpo-sft ./output/checkpoints/2024-06-10_12-05-36
```