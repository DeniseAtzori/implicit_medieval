# ============================================================
# Benchmark — classification de séquences pour la détection
# d'enthymèmes (AutoModelForSequenceClassification + LoRA/peft)
#
# Points clés vs approche générative (Unsloth) :
#   • Sortie directe : 2 logits → argmax (pas de génération texte)
#   • compute_metrics() natif HF → métriques à chaque epoch
#   • Sélection automatique du meilleur checkpoint (F1-macro)
#   • max_length=512 (tweet seul, pas de prompt)
#   • Gestion du déséquilibre de classes (datasets extra = enthymeme only)
# ============================================================

import os
import time
import gc
import csv
import random

import torch
import numpy as np

from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. MODÈLES À TESTER
#    — encodeurs BERT-like (idéaux pour la classification)
#    — décodeurs légers (pour comparaison avec l'approche générative)
# ============================================================

models_to_test = [

    # ── Encodeurs ──────────────────────────────────────────
    # RoBERTa : référence solide, robuste sur les textes courts
    "FacebookAI/roberta-base",

    # DeBERTa-v3 : souvent meilleur que RoBERTa sur NLU
    "microsoft/deberta-v3-base",

    # BERT multilingue : utile si les tweets sont mixtes
    "google-bert/bert-base-multilingual-cased",

    # DistilBERT : léger, bon rapport perf/vitesse
    "distilbert/distilbert-base-uncased",

    # ELECTRA : pré-entraîné avec discrimination (souvent meilleur que BERT)
    "google/electra-base-discriminator",

    # XLM-RoBERTa : multilingue, fort sur les réseaux sociaux
    "FacebookAI/xlm-roberta-base",

    # ── Décodeurs légers ───────────────────────────────────
    # Qwen2.5-0.5B : très léger, bon point de comparaison
    # "Qwen/Qwen2.5-0.5B",

    # Llama-3.2-1B : comparaison directe avec l'approche Unsloth
    # "meta-llama/Llama-3.2-1B",
]


# ============================================================
# 2. CHEMINS DES DATASETS
# ============================================================

DATASET_DIR = "datasets_prepared"

INITIAL_FILE = "dataset_initial_corrected.json"

EXTRA_FILES = [
    "arct_binary_corrected.json",
    "implicit_hate_binary_corrected.json",
    "external_combined_binary_corrected.json",
]

dataset_base_path = os.path.join(DATASET_DIR, INITIAL_FILE)

dataset_extra_paths = [
    os.path.join(DATASET_DIR, filename)
    for filename in EXTRA_FILES
]

# ============================================================
# 3. PARAMÈTRES GÉNÉRAUX
# ============================================================

# 512 couvre les tweets jusqu'à ~801 tokens tout en économisant
# de la mémoire vs 1024. Les rares tweets tronqués ne perdent
# que leur toute fin (peu discriminante pour les enthymèmes).
MAX_SEQ_LENGTH = 512

BENCHMARK_DIR = "benchmark_enthymeme_seqclf"
os.makedirs(BENCHMARK_DIR, exist_ok=True)

CSV_PATH = os.path.join(BENCHMARK_DIR, "benchmark_metrics.csv")

CSV_HEADERS = [
    "model_name",
    "dataset_tag",
    "balance_strategy",     # none / class_weight / undersample / oversample
    "status",
    "train_loss",
    "eval_loss",
    "accuracy",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "n_train",              # taille réelle du dataset d'entraînement
    "n_eval",
    "ratio_implicit_train", # % d'implicit dans le train set
    "train_runtime_sec",
    "output_dir",
    "error",
]

LABEL2ID = {"none": 0, "implicit": 1}
ID2LABEL = {0: "none", 1: "implicit"}

# ── Stratégie d'équilibrage des classes ────────────────────
#
# Problème : les datasets extra contiennent uniquement des
# implicit → déséquilibre artificiel → sur-prédiction de
# la classe implicit → recall "none" s'effondre.
#
# Options :
#   "none"         — aucun rééquilibrage (baseline)
#   "class_weight" — poids inversement proportionnels aux fréquences
#                    dans la loss (recommandé : pas de perte de données)
#   "undersample"  — réduction de la classe majoritaire
#   "oversample"   — duplication de la classe minoritaire
#
# Appliqué UNIQUEMENT aux datasets combinés (phase 2).
# La phase 1 (dataset de base) utilise toujours "none".
BALANCE_STRATEGY = "class_weight"


# ============================================================
# 4. CONFIGURATION LoRA
# ============================================================

def get_lora_config(model_name: str) -> LoraConfig:
    """
    Retourne une LoraConfig adaptée à l'architecture.

    Encodeurs BERT-like  → couches 'query' / 'value'
    Décodeurs (Llama…)   → couches 'q_proj' / 'v_proj'

    modules_to_save garantit que la tête de classification
    (ajoutée par HF) est aussi entraînée — sans ça le modèle
    n'apprendrait rien de la tâche.
    """
    name = model_name.lower()
    is_decoder = any(k in name for k in [
        "llama", "qwen", "mistral", "falcon", "gpt", "bloom", "opt", "phi"
    ])

    target_modules = ["q_proj", "v_proj"] if is_decoder else ["query", "value"]

    # DeBERTa utilise des noms de couches différents
    if "deberta" in name:
        target_modules = ["query_proj", "value_proj"]

    # ELECTRA utilise aussi des noms spécifiques
    if "electra" in name:
        target_modules = ["query", "value"]

    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=target_modules,
        # La tête de classification doit être entièrement entraînable
        modules_to_save=["classifier", "score", "out_proj"],
    )


# ============================================================
# 5. ÉQUILIBRAGE DES CLASSES
# ============================================================

def get_class_distribution(dataset_raw) -> dict:
    counts = {"implicit": 0, "none": 0}

    for example in dataset_raw:
        lbl = str(example["categorie"]).strip().lower()
        if lbl == "implicit":
            counts["implicit"] += 1
        else:
            counts["none"] += 1

    return counts

def balance_dataset(dataset_raw, strategy: str, seed: int = 42):
    """
    Applique la stratégie d'équilibrage choisie sur le dataset brut.

    Paramètres
    ----------
    dataset_raw : dataset HuggingFace avant tokenisation
    strategy    : "none" | "class_weight" | "undersample" | "oversample"
    seed        : graine pour la reproductibilité

    Retourne
    --------
    (dataset_équilibré, class_weights_ou_None)
        class_weights : tenseur [w_none, w_enthymeme] si strategy="class_weight"
                        None sinon
    """
    if strategy == "none":
        return dataset_raw, None

    # Séparation par classe
    examples = list(dataset_raw)
    enthymemes = [e for e in examples if str(e["categorie"]).lower() == "enthymeme"]
    nones      = [e for e in examples if str(e["categorie"]).lower() != "enthymeme"]

    n_enth = len(enthymemes)
    n_none = len(nones)

    print(f"\n  Distribution avant équilibrage : "
          f"enthymeme={n_enth}, none={n_none} "
          f"(ratio={n_enth/(n_enth+n_none):.1%})")

    if strategy == "class_weight":
        # Pas de modification du dataset — on calcule les poids
        # qui seront passés à la loss function dans le Trainer
        all_labels = [LABEL2ID["enthymeme"]] * n_enth + [LABEL2ID["none"]] * n_none
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=np.array(all_labels),
        )
        class_weights = torch.tensor(weights, dtype=torch.float)
        print(f"  Poids de classe : none={weights[0]:.3f}, enthymeme={weights[1]:.3f}")
        return dataset_raw, class_weights

    elif strategy == "undersample":
        # Réduction de la classe majoritaire
        rng = random.Random(seed)
        n_target = min(n_enth, n_none)
        if n_enth > n_none:
            enthymemes = rng.sample(enthymemes, n_target)
        else:
            nones = rng.sample(nones, n_target)
        balanced = Dataset.from_list(enthymemes + nones)
        print(f"  Après undersample : {len(balanced)} exemples "
              f"({n_target} par classe)")
        return balanced, None

    elif strategy == "oversample":
        # Duplication de la classe minoritaire
        rng = random.Random(seed)
        n_target = max(n_enth, n_none)
        if n_enth < n_none:
            extras = rng.choices(enthymemes, k=n_target - n_enth)
            enthymemes = enthymemes + extras
        else:
            extras = rng.choices(nones, k=n_target - n_none)
            nones = nones + extras
        balanced = Dataset.from_list(enthymemes + nones)
        print(f"  Après oversample : {len(balanced)} exemples "
              f"({n_target} par classe)")
        return balanced, None

    else:
        raise ValueError(f"Stratégie inconnue : {strategy}")


# ============================================================
# 6. TRAINER AVEC POIDS DE CLASSE
# ============================================================

class WeightedTrainer(Trainer):
    """
    Trainer HuggingFace étendu pour supporter les poids de classe.

    Surcharge compute_loss() pour appliquer CrossEntropyLoss
    pondérée — la loss sur la classe minoritaire est multipliée
    par son poids, forçant le modèle à y prêter plus d'attention.

    Utilisé uniquement si strategy="class_weight".
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        # CrossEntropyLoss pondérée
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ============================================================
# 7. FONCTIONS UTILITAIRES
# ============================================================

def convert_label(category: str) -> int:
    """Convertit la catégorie textuelle en entier 0/1."""
    if category is not None and str(category).strip().lower() == "implicit":
        return LABEL2ID["implicit"]
    return LABEL2ID["none"]

def write_csv_row(row: dict) -> None:
    """Ajoute une ligne au CSV — écriture immédiate (robustesse aux crashs)."""
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def free_memory(*objects) -> None:
    """Libère la mémoire GPU — appelé dans finally après chaque run."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def tokenize_dataset(dataset_raw, tokenizer):
    """
    Tokenise le dataset brut pour le Trainer HF.

    Pas de prompt template — le tweet est tokenisé directement.
    Le padding dynamique est géré par DataCollatorWithPadding.
    """
    def _tokenize(examples):
        encoded = tokenizer(
            [str(t).strip() for t in examples["tweet_text"]],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        encoded["label"] = [convert_label(c) for c in examples["categorie"]]
        return encoded

    tokenized = dataset_raw.map(_tokenize, batched=True)

    cols_to_remove = [
        c for c in dataset_raw.column_names
        if c not in ["input_ids", "attention_mask", "label", "token_type_ids"]
    ]
    tokenized = tokenized.remove_columns(cols_to_remove)
    tokenized.set_format("torch")
    return tokenized


def make_compute_metrics():
    """
    Retourne la fonction compute_metrics pour le Trainer.
    Appelée automatiquement à chaque epoch d'évaluation.
    """
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        labels_str = [ID2LABEL[i] for i in labels]
        preds_str  = [ID2LABEL[i] for i in preds]

        print("\n  Rapport de classification :")
        print(classification_report(
            labels_str, preds_str,
            target_names=list(LABEL2ID.keys()),
            zero_division=0,
        ))

        return {
            "accuracy":        round(accuracy_score(labels, preds), 4),
            "precision_micro": round(precision_score(labels, preds, average="micro",  zero_division=0), 4),
            "recall_micro":    round(recall_score   (labels, preds, average="micro",  zero_division=0), 4),
            "f1_micro":        round(f1_score        (labels, preds, average="micro",  zero_division=0), 4),
            "precision_macro": round(precision_score(labels, preds, average="macro",  zero_division=0), 4),
            "recall_macro":    round(recall_score   (labels, preds, average="macro",  zero_division=0), 4),
            "f1_macro":        round(f1_score        (labels, preds, average="macro",  zero_division=0), 4),
        }

    return compute_metrics


def build_training_args(output_dir: str) -> TrainingArguments:
    """
    Hyperparamètres pour la classification de séquences.

    LR 2e-5 (vs 2e-4 génératif) : les encodeurs sont
    plus sensibles aux grands learning rates.
    Sélection du meilleur checkpoint sur F1-macro.
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        optim="adamw_torch",
        seed=42,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
    )


# ============================================================
# 8. FONCTION PRINCIPALE D'ENTRAÎNEMENT
# ============================================================

def train_one_model(
    model_name:       str,
    dataset_tag:      str,
    dataset_raw,
    apply_balancing:  bool = False,
) -> dict:
    """
    Cycle complet pour un couple (modèle, dataset).

    Paramètres
    ----------
    model_name      : identifiant HuggingFace
    dataset_tag     : étiquette pour le CSV / les dossiers
    dataset_raw     : dataset HuggingFace brut (non splitté)
    apply_balancing : True pour les datasets combinés (phase 2)
                      False pour le dataset de base seul (phase 1)
    """

    print(f"\n{'='*80}")
    print(f"  Modèle    : {model_name}")
    print(f"  Dataset   : {dataset_tag}")
    print(f"  Équilibre : {'oui → ' + BALANCE_STRATEGY if apply_balancing else 'non (dataset de base)'}")
    print(f"{'='*80}")

    start_time = time.time()

    safe_model   = model_name.replace("/", "__")
    safe_dataset = dataset_tag.replace("+", "_plus_")
    run_dir      = os.path.join(BENCHMARK_DIR, safe_model, safe_dataset)
    os.makedirs(run_dir, exist_ok=True)

    final_model_dir = os.path.join(run_dir, "final_model")

    csv_row = {h: None for h in CSV_HEADERS}
    csv_row.update({
        "model_name":        model_name,
        "dataset_tag":       dataset_tag,
        "balance_strategy":  BALANCE_STRATEGY if apply_balancing else "none",
        "status":            "failed",
        "output_dir":        final_model_dir,
    })

    model     = None
    tokenizer = None
    trainer   = None

    try:
        # ── Équilibrage des classes ────────────────────────
        if apply_balancing:
            dataset_balanced, class_weights = balance_dataset(
                dataset_raw, strategy=BALANCE_STRATEGY
            )
        else:
            dataset_balanced = dataset_raw
            class_weights    = None

        # ── Diagnostic de distribution ─────────────────────
        dist = get_class_distribution(dataset_balanced)
        n_total = dist["enthymeme"] + dist["none"]
        ratio_enth = dist["enthymeme"] / n_total if n_total > 0 else 0
        print(f"  Distribution : enthymeme={dist['enthymeme']}, "
              f"none={dist['none']} (ratio={ratio_enth:.1%})")

        # ── Chargement du tokenizer ────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # ── Chargement du modèle de classification ─────────
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        # ── Application de LoRA ────────────────────────────
        lora_cfg = get_lora_config(model_name)
        model    = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        # ── Tokenisation + split ───────────────────────────
        tokenized = tokenize_dataset(dataset_balanced, tokenizer)
        split     = tokenized.train_test_split(test_size=0.2, seed=42)

        dataset_train = split["train"]
        dataset_eval  = split["test"]

        csv_row["n_train"]              = len(dataset_train)
        csv_row["n_eval"]               = len(dataset_eval)
        csv_row["ratio_enthymeme_train"] = round(ratio_enth, 4)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        training_args = build_training_args(run_dir)

        # ── Choix du Trainer selon la stratégie ───────────
        # WeightedTrainer si class_weight, Trainer standard sinon
        TrainerClass = WeightedTrainer if class_weights is not None else Trainer

        trainer = TrainerClass(
            model=model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_eval,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=make_compute_metrics(),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            **({"class_weights": class_weights} if class_weights is not None else {}),
        )

        # ── Entraînement ───────────────────────────────────
        train_output  = trainer.train()
        final_metrics = trainer.evaluate()

        # ── Sauvegarde ─────────────────────────────────────
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        elapsed = time.time() - start_time

        csv_row.update({
            "status":            "success",
            "train_loss":        round(train_output.training_loss, 6),
            "eval_loss":         round(final_metrics.get("eval_loss", 0.0), 6),
            "accuracy":          final_metrics.get("eval_accuracy"),
            "precision_micro":   final_metrics.get("eval_precision_micro"),
            "recall_micro":      final_metrics.get("eval_recall_micro"),
            "f1_micro":          final_metrics.get("eval_f1_micro"),
            "precision_macro":   final_metrics.get("eval_precision_macro"),
            "recall_macro":      final_metrics.get("eval_recall_macro"),
            "f1_macro":          final_metrics.get("eval_f1_macro"),
            "train_runtime_sec": round(elapsed, 1),
        })

        print(f"\n  ✓ F1-macro  = {csv_row['f1_macro']}")
        print(f"  ✓ Accuracy  = {csv_row['accuracy']}")
        print(f"  ✓ Sauvegardé → {final_model_dir}")

    except Exception as e:
        elapsed = time.time() - start_time
        csv_row["train_runtime_sec"] = round(elapsed, 1)
        csv_row["error"] = str(e)
        print(f"\n  ✗ Erreur : {e}")

    finally:
        free_memory(model, tokenizer, trainer)

    write_csv_row(csv_row)
    return csv_row


# ============================================================
# 9. CHARGEMENT DES DATASETS
# ============================================================

print("Chargement des datasets…")

dataset_base = load_dataset(
    "json", data_files=dataset_base_path, split="train"
)
dist_base = get_class_distribution(dataset_base)
print(f"  Dataset de base : {len(dataset_base)} exemples  "
      f"(enthymeme={dist_base['enthymeme']}, none={dist_base['none']})")

dataset_extras = []
for i, path in enumerate(dataset_extra_paths, start=1):
    ds = load_dataset("json", data_files=path, split="train")
    dist_extra = get_class_distribution(ds)
    dataset_extras.append(ds)
    print(f"  Dataset extra #{i}: {len(ds)} exemples  "
          f"(enthymeme={dist_extra['enthymeme']}, none={dist_extra['none']})")
    if dist_extra["none"] == 0:
        print(f"  ⚠  Dataset extra #{i} contient uniquement des enthymèmes "
              f"→ stratégie '{BALANCE_STRATEGY}' sera appliquée en phase 2")


# ============================================================
# 10. PHASE 1 — TOUS LES MODÈLES SUR LE DATASET DE BASE
# ============================================================

print("\n" + "="*80)
print("PHASE 1 : Tous les modèles sur le dataset de base (sans rééquilibrage)")
print("="*80)

phase1_results = []

for model_name in models_to_test:
    result = train_one_model(
        model_name=model_name,
        dataset_tag="base",
        dataset_raw=dataset_base,
        apply_balancing=False,  # dataset de base déjà équilibré
    )
    phase1_results.append(result)


# ============================================================
# 11. SÉLECTION DU MEILLEUR MODÈLE
# ============================================================

successful = [r for r in phase1_results if r["status"] == "success"]

if not successful:
    print("\n⚠  Tous les modèles ont échoué. Phase 2 ignorée.")
    best_model_name = None
else:
    best = max(
        successful,
        key=lambda r: (
            float(r["f1_macro"] or 0),
            -float(r["eval_loss"] or 9999),
        ),
    )
    best_model_name = best["model_name"]
    print(
        f"\n🏆 Meilleur modèle : {best_model_name}"
        f"  (F1-macro={best['f1_macro']}, accuracy={best['accuracy']})"
    )


# ============================================================
# 12. PHASE 2 — MEILLEUR MODÈLE × DATASETS SUPPLÉMENTAIRES
#     Rééquilibrage activé car les datasets extra sont enthymeme-only
# ============================================================

if best_model_name and dataset_extras:
    print("\n" + "="*80)
    print(f"PHASE 2 : {best_model_name} + datasets étendus")
    print("="*80)

    for path, ds_extra in zip(dataset_extra_paths, dataset_extras):
        extra_name = os.path.basename(path).replace("_corrected.json", "")
        dataset_tag = f"initial+{extra_name}"

        combined = concatenate_datasets([dataset_base, ds_extra])

        print(
            f"\n  Dataset '{dataset_tag}' : "
            f"{len(dataset_base)} + {len(ds_extra)} = {len(combined)} exemples"
        )

        train_one_model(
            model_name=best_model_name,
            dataset_tag=dataset_tag,
            dataset_raw=combined,
            apply_balancing=False
        )

elif not dataset_extras:
    print("\nℹ  Aucun dataset supplémentaire — phase 2 ignorée.")

# ============================================================
# 13. RÉCAPITULATIF FINAL
# ============================================================

print("\n" + "="*80)
print("BENCHMARK TERMINÉ")
print("="*80)
print(f"\nMétriques → {CSV_PATH}\n")

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

COL = {"model": 38, "tag": 16, "bal": 14, "status": 10, "metric": 10}
header_line = (
    f"{'model_name':<{COL['model']}} "
    f"{'dataset_tag':<{COL['tag']}} "
    f"{'balance':<{COL['bal']}} "
    f"{'status':<{COL['status']}} "
    f"{'accuracy':<{COL['metric']}} "
    f"{'f1_macro':<{COL['metric']}} "
    f"{'f1_micro':<{COL['metric']}} "
    f"{'eval_loss':<{COL['metric']}}"
)
print(header_line)
print("-" * len(header_line))

for row in rows:
    print(
        f"{row['model_name']:<{COL['model']}} "
        f"{row['dataset_tag']:<{COL['tag']}} "
        f"{str(row.get('balance_strategy','')):<{COL['bal']}} "
        f"{row['status']:<{COL['status']}} "
        f"{str(row.get('accuracy','')):<{COL['metric']}} "
        f"{str(row.get('f1_macro','')):<{COL['metric']}} "
        f"{str(row.get('f1_micro','')):<{COL['metric']}} "
        f"{str(row.get('eval_loss','')):<{COL['metric']}}"
    )