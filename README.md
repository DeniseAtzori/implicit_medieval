# Folder for the MediaEval task: Missing pieces and misinformation

Shared Task link: (https://multimediaeval.github.io/editions/2026/tasks/enthymeme/)

## Task Description
Given a tweet, determine whether it contains an implicit premise, an implicit conclusion, or neither. This is a three-class classification task.

- Input: The raw text of a tweet.
- Output: One label: implicit_premise, implicit_conclusion, or none.

## Task 1: “Enthymeme Detection”
Detecting the absence or presence of enthymemes in tweets (three-class classification)

### Constrained Run 1
Predict the label from the tweet text alone. No external data or additional annotation information is permitted.

**Tried and tested approaches:**

- Comparison of two training strategies for DeBERTa-v3-base/large: full fine-tuning and LoRA, using hyperparameter grid search

### Constrained Run 2
In addition to the tweet text, use the raw labels provided by three independent annotators. The goal is to investigate whether modelling annotator disagreement improves performance, especially on borderline cases. The output label is the same three-class prediction.

**Tried and tested approaches:**

- Full fine-tuning and LoRA for DeBERTa-v3-large with KL Divergence Loss (with ablation on alpha)
- Fine-tuning Qwen2.5-3B-Instruct-bnb-4bit with Unsloth

## Task 2: Proposition Generation”
For each tweet classified as containing an implicit argument, generate the text of the missing proposition. Task 2 requires prior completion of Task 1, as the predicted label is part of the input.

- Fine-tuning Qwen2.5-3B-Instruct-bnb-4bit with Unsloth