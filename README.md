# Folder for the MediaEval task: Missing pieces and misinformation

Shared Task link: (https://multimediaeval.github.io/editions/2026/tasks/enthymeme/)

## Task Description
Given a tweet, determine whether it contains an implicit premise, an implicit conclusion, or neither. This is a three-class classification task.

- Input: The raw text of a tweet.
- Output: One label: implicit_premise, implicit_conclusion, or none.

## Task 1: “Enthymeme Detection” — Detecting the absence or presence of enthymemes in tweets (three-class classification)

### Constrained Run 1
Predict the label from the tweet text alone. No external data or additional annotation information is permitted.

**Tried and tested approaches:**

- Comparison of two training strategies for DeBertA-v3-base: full fine-tuning and LoRA, using hyperparameter grid search