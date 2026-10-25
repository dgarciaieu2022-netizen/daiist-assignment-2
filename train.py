"""
Assignment 2: training pipeline (run with `uv run python main.py train`).

Start to finish, with no manual steps:
  1. Download the dataset version named in config.py from Roboflow Universe.
  2. Describe it: images and instances per class and split, object sizes.
  3. Finetune yolo11s from its COCO checkpoint; copy the best weights to weights/.
  4. Evaluate on the test split, per class, against two baselines:
     the COCO model zero-shot, and trivial predict-nothing / majority-class.
  5. Score every validation and test egg, sweep the threshold, cost it with
     the figures in config.py, and pick the cheapest operating point
     (chosen on validation, reported on test).
  6. Save everything app.py needs into artifacts/, so the app never trains.
"""

import json
import random
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

import config
import decision
from api_key import load_api_key

SPLITS = {"train": "train", "valid": "valid", "test": "test"}
CM_CONF = 0.25  # confidence used only for the box-level confusion matrix
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def log(msg: str) -> None:
    print(f"\n[train] {msg}", flush=True)


def save_json(obj, name: str) -> None:
    (config.ARTIFACTS / name).write_text(json.dumps(obj, indent=2, default=float))


# --- 1. Data ------------------------------------------------------------------

def download_dataset() -> None:
    if (config.DATASET_DIR / "data.yaml").exists():
        log(f"Dataset already downloaded at {config.DATASET_DIR}")
        return
    from roboflow import Roboflow

    log(f"Downloading {config.RF_WORKSPACE}/{config.RF_PROJECT} v{config.RF_VERSION}")
    rf = Roboflow(api_key=load_api_key())
    (rf.workspace(config.RF_WORKSPACE)
       .project(config.RF_PROJECT)
       .version(config.RF_VERSION)
       .download(config.RF_FORMAT, location=str(config.DATASET_DIR), overwrite=True))


def write_data_yaml() -> tuple[Path, list[str]]:
    """Roboflow's data.yaml uses relative paths that break depending on where
    YOLO is launched from. Write one with absolute paths instead."""
    original = yaml.safe_load((config.DATASET_DIR / "data.yaml").read_text())
    names = list(original["names"].values()) if isinstance(original["names"], dict) else list(original["names"])
    fixed = {
        "path": str(config.DATASET_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    path = config.DATASET_DIR / "data_abs.yaml"
    path.write_text(yaml.safe_dump(fixed, sort_keys=False))
    return path, names


def images_in(split: str) -> list[Path]:
    folder = config.DATASET_DIR / SPLITS[split] / "images"
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def read_labels(image: Path) -> np.ndarray:
    """YOLO label rows (cls, cx, cy, w, h), normalised. Empty array if no boxes."""
    label = image.parent.parent / "labels" / f"{image.stem}.txt"
    if not label.exists() or not label.read_text().strip():
        return np.zeros((0, 5))
    return np.loadtxt(label, ndmin=2)[:, :5]


# --- 2. Describe the dataset ----------------------------------------------------

def describe_dataset(names: list[str], pos_idx: int) -> dict:
    log("Describing the dataset")
    rows, boxes = [], []
    for split in SPLITS:
        for img in images_in(split):
            lab = read_labels(img)
            classes = set(lab[:, 0].astype(int)) if len(lab) else set()
            rows.append({
                "split": split, "image": img.name, "n_boxes": len(lab),
                "has_cracked": pos_idx in classes,
                "has_uncracked": any(c != pos_idx for c in classes),
            })
            for c, _, _, w, h in lab:
                boxes.append({"split": split, "class": names[int(c)],
                              "area_frac": w * h, "width_frac": w, "height_frac": h})
    imgs, bx = pd.DataFrame(rows), pd.DataFrame(boxes)
    bx.to_csv(config.ARTIFACTS / "box_sizes.csv", index=False)

    stats = {"images_per_split": imgs.groupby("split").size().to_dict()}
    stats["instances_per_class_and_split"] = (
        bx.groupby(["split", "class"]).size().unstack(fill_value=0).to_dict(orient="index"))
    stats["images_with_cracked_per_split"] = imgs.groupby("split")["has_cracked"].sum().astype(int).to_dict()
    stats["images_with_no_boxes"] = imgs.groupby("split")["n_boxes"].apply(lambda s: int((s == 0).sum())).to_dict()
    stats["images_with_both_classes"] = int((imgs.has_cracked & imgs.has_uncracked).sum())
    stats["boxes_per_image"] = imgs["n_boxes"].describe().round(2).to_dict()
    stats["box_area_fraction_by_class"] = (
        bx.groupby("class")["area_frac"].describe(percentiles=[.1, .25, .5, .75, .9]).round(4)
          .to_dict(orient="index"))
    stats["small_boxes_share_under_1pct_of_image"] = round(float((bx.area_frac < 0.01).mean()), 4)
    save_json(stats, "dataset_stats.json")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = bx.groupby(["split", "class"]).size().unstack(fill_value=0).loc[list(SPLITS)]
    counts.plot.bar(ax=axes[0], rot=0, title="Instances per class and split")
    for cls, grp in bx.groupby("class"):
        axes[1].hist(np.sqrt(grp.area_frac), bins=30, alpha=0.6, label=cls)
    axes[1].set(title="Box size: sqrt(box area / image area)", xlabel="relative side length")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(config.ARTIFACTS / "dataset_overview.png", dpi=120)
    plt.close(fig)
    print(json.dumps(stats, indent=2, default=float))
    return stats


# --- 3. Finetune ------------------------------------------------------------------

def pick_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def base_checkpoint() -> str:
    """Download the COCO checkpoint into .models/ so it never lands in weights/."""
    config.MODELS_CACHE.mkdir(exist_ok=True)
    return str(config.MODELS_CACHE / config.BASE_MODEL)


def finetune(data_yaml: Path) -> YOLO:
    device = pick_device()
    log(f"Finetuning {config.BASE_MODEL} on device={device}")
    model = YOLO(base_checkpoint())
    model.train(
        data=str(data_yaml), epochs=config.EPOCHS, patience=config.PATIENCE,
        imgsz=config.IMGSZ, batch=config.BATCH, seed=config.SEED, device=device,
        project=str(config.RUNS_DIR), name="finetune", exist_ok=True, plots=True,
        workers=2,
    )
    best = config.RUNS_DIR / "finetune" / "weights" / "best.pt"
    config.WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    shutil.copy(best, config.WEIGHTS_PATH)
    for extra in ["results.png", "results.csv"]:
        src = config.RUNS_DIR / "finetune" / extra
        if src.exists():
            shutil.copy(src, config.ARTIFACTS / f"training_{extra}")
    log(f"Best weights copied to {config.WEIGHTS_PATH}")
    return YOLO(str(config.WEIGHTS_PATH))


# --- 4. Evaluate on test ------------------------------------------------------------

def detection_metrics(model: YOLO, data_yaml: Path, names: list[str]) -> dict:
    log("Detection metrics on the test split")
    m = model.val(data=str(data_yaml), split="test", imgsz=config.IMGSZ, device=pick_device(),
                  project=str(config.RUNS_DIR), name="test_eval", exist_ok=True, plots=True)
    per_class = []
    for i, cls_idx in enumerate(m.box.ap_class_index):
        per_class.append({
            "class": names[int(cls_idx)],
            "precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
            "AP50": float(m.box.ap50[i]), "AP50-95": float(m.box.ap[i]),
        })
    shutil.copy(config.RUNS_DIR / "test_eval" / "BoxPR_curve.png", config.ARTIFACTS / "test_BoxPR_curve.png")

    # AP needs every detection (conf=0.001), but a confusion matrix built from
    # those is flooded with near-zero-confidence background boxes. Build the
    # matrix at a readable confidence of 0.25 instead.
    cm_run = model.val(data=str(data_yaml), split="test", imgsz=config.IMGSZ, device=pick_device(),
                       conf=CM_CONF, project=str(config.RUNS_DIR), name="test_eval_cm",
                       exist_ok=True, plots=True)
    matrix = cm_run.confusion_matrix.matrix.astype(int).tolist()  # rows = predicted, cols = true
    for name in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        src = config.RUNS_DIR / "test_eval_cm" / name
        if src.exists():
            shutil.copy(src, config.ARTIFACTS / f"test_{name}")
    return {
        "overall": {"precision": float(m.box.mp), "recall": float(m.box.mr),
                    "AP50": float(m.box.map50), "mAP50-95": float(m.box.map)},
        "per_class": per_class,
        "confusion_matrix": {"rows_predicted_cols_true": matrix, "labels": names + ["background"],
                             "confidence": CM_CONF},
        "note": "Per-class precision/recall are at the confidence that maximises mean F1 (Ultralytics).",
    }


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gt_boxes_xyxy(image: Path, shape) -> list:
    h, w = shape
    return [((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            for _, cx, cy, bw, bh in read_labels(image)]


def egg_found_rate(model: YOLO, images: list[Path], ignore: set[str]) -> tuple[float, dict]:
    """Share of test images where the model puts any box (whatever its class) on
    the labelled egg with IoU >= 0.5, plus which class names it used."""
    hits, used = 0, {}
    for res in model.predict([str(p) for p in images], conf=0.25, imgsz=config.IMGSZ,
                             device=pick_device(), stream=True, verbose=False):
        img = next(p for p in images if p.name == Path(res.path).name)
        gts = gt_boxes_xyxy(img, res.orig_shape)
        found = False
        for box, cls in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist()):
            name = res.names[int(cls)]
            if name in ignore:
                continue
            used[name] = used.get(name, 0) + 1
            if any(iou(box, g) >= 0.5 for g in gts):
                found = True
        hits += found
    return hits / max(len(images), 1), dict(sorted(used.items(), key=lambda kv: -kv[1]))


def score_images(model: YOLO, split: str, pos_idx: int) -> pd.DataFrame:
    """Score = highest 'cracked egg' confidence in the image (0 if none).
    Truth = the image contains a labelled cracked egg."""
    imgs = images_in(split)
    rows = []
    for res in model.predict([str(p) for p in imgs], conf=0.001, imgsz=config.IMGSZ,
                             device=pick_device(), stream=True, verbose=False):
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        img = Path(res.path)
        lab = read_labels(img)
        rows.append({
            "split": split, "image": img.name,
            "label": int(len(lab) > 0 and pos_idx in set(lab[:, 0].astype(int))),
            "score": float(conf[cls == pos_idx].max()) if np.any(cls == pos_idx) else 0.0,
            "score_uncracked": float(conf[cls != pos_idx].max()) if np.any(cls != pos_idx) else 0.0,
        })
    return pd.DataFrame(rows)


def image_level(decisions_fail: np.ndarray, labels: np.ndarray) -> dict:
    tp = int(np.sum(decisions_fail & (labels == 1)))
    fp = int(np.sum(decisions_fail & (labels == 0)))
    fn = int(np.sum(~decisions_fail & (labels == 1)))
    tn = int(np.sum(~decisions_fail & (labels == 0)))
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": (tp + tn) / max(len(labels), 1),
        "recall_cracked": tp / max(tp + fn, 1),
        "precision_cracked": tp / max(tp + fp, 1) if tp + fp else 0.0,
        "false_fail_rate": fp / max(fp + tn, 1),
    }


def baselines(finetuned: YOLO, scores_test: pd.DataFrame, chosen_t: float,
              names: list[str], costs: decision.Costs) -> dict:
    log("Baselines: COCO zero-shot and trivial")
    test_imgs = images_in("test")
    labels = scores_test["label"].to_numpy()

    coco = YOLO(base_checkpoint())
    coco_found, coco_classes = egg_found_rate(coco, test_imgs, ignore={"person"})
    ft_found, _ = egg_found_rate(finetuned, test_imgs, ignore=set())
    coco_has_our_classes = [n for n in names if n in coco.names.values()]

    train_lab = [int(len(l) > 0 and POS_IDX in set(l[:, 0].astype(int)))
                 for l in (read_labels(p) for p in images_in("train"))]
    majority_is_cracked = np.mean(train_lab) > 0.5

    def costed(fail_mask):
        r = decision.conditional_rates(fail_mask.astype(float), labels, 0.5, 0.5)
        return 1000 * decision.cost_per_egg(r, costs, config.BASE_RATE)

    ft_fail = scores_test["score"].to_numpy() >= chosen_t
    nothing = np.zeros(len(labels), dtype=bool)
    majority = np.full(len(labels), majority_is_cracked)
    return {
        "detection_AP50_by_model": {
            "finetuned yolo11s": "see detection_metrics.json",
            "COCO yolo11s zero-shot": 0.0,
            "predict nothing": 0.0,
        },
        "coco_zero_shot": {
            "our_classes_in_coco_vocabulary": coco_has_our_classes,
            "share_of_test_eggs_it_boxes_IoU50_any_class": coco_found,
            "classes_it_used_on_test_images": coco_classes,
            "why_AP_is_zero": "COCO has no 'egg' or 'cracked' class, so it cannot emit "
                              "either label; at best it boxes the egg under another name.",
        },
        "finetuned_share_of_test_eggs_boxed_IoU50": ft_found,
        "majority_class_in_train_images": "cracked egg" if majority_is_cracked else "Uncracked egg",
        "image_level_decision_on_test": {
            "finetuned (single threshold)": {**image_level(ft_fail, labels), "cost_per_1000": costed(ft_fail)},
            "COCO zero-shot (cannot say cracked -> passes all)": {**image_level(nothing, labels), "cost_per_1000": costed(nothing)},
            "predict nothing (pass all)": {**image_level(nothing, labels), "cost_per_1000": costed(nothing)},
            "majority class": {**image_level(majority, labels), "cost_per_1000": costed(majority)},
        },
    }


# --- 5. Cost sweep and operating point ---------------------------------------------

def bootstrap_ci(scores, labels, t_low, t_high, costs, n=1000, seed=0) -> dict:
    """Resample the test eggs (within each class) to see how much the cost would
    move with a different 79 images. Wide interval = do not over-read the point."""
    rng = np.random.default_rng(seed)
    scores, labels = np.asarray(scores), np.asarray(labels)
    pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(decision.evaluate(scores[idx], labels[idx], t_low, t_high, costs, config.BASE_RATE)["cost_per_1000"])
    return {"p2.5": float(np.percentile(vals, 2.5)), "p97.5": float(np.percentile(vals, 97.5))}


def sensitivity(scores, labels, costs: decision.Costs) -> pd.DataFrame:
    """How the cheapest single threshold moves if the two most uncertain inputs
    (cost of a missed crack, base rate) are wrong."""
    rows = []
    for fn in [0.5, 1, 2, 3, 5, 10, 20]:
        for br in [0.01, 0.02, 0.05, 0.10, 0.20]:
            c = decision.Costs(**{**costs.__dict__, "fn": fn})
            best = decision.best_single(decision.sweep_single(scores, labels, c, br))
            rows.append({"cost_fn": fn, "base_rate": br, "best_threshold": best["threshold"],
                         "cost_per_1000": round(best["cost_per_1000"], 2),
                         "recall": round(best["recall"], 3)})
    return pd.DataFrame(rows)


def plot_cost_curve(sweep_val, sweep_test, t_star) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sweep_val.threshold, sweep_val.cost_per_1000, label="validation (used to choose)")
    ax.plot(sweep_test.threshold, sweep_test.cost_per_1000, label="test (reported)", alpha=0.8)
    ax.axvline(t_star, color="k", ls="--", label=f"chosen threshold = {t_star:.2f}")
    ax.set(xlabel="confidence threshold for 'cracked egg'", ylabel="expected cost per 1,000 eggs (EUR)",
           title=f"Cost vs threshold (base rate {config.BASE_RATE:.0%}, FN EUR {config.COST_FN}, FP EUR {config.COST_FP})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.ARTIFACTS / "cost_curve.png", dpi=120)
    plt.close(fig)


def operating_point(model: YOLO, names: list[str]) -> tuple[pd.DataFrame, dict, decision.Costs]:
    log("Scoring validation and test eggs, sweeping thresholds")
    costs = decision.Costs()
    scores = pd.concat([score_images(model, "valid", POS_IDX), score_images(model, "test", POS_IDX)])
    scores.to_csv(config.ARTIFACTS / "image_scores.csv", index=False)
    val, test = scores[scores.split == "valid"], scores[scores.split == "test"]

    sweep_val = decision.sweep_single(val.score, val.label, costs, config.BASE_RATE)
    sweep_test = decision.sweep_single(test.score, test.label, costs, config.BASE_RATE)
    pd.concat([sweep_val.assign(split="valid"), sweep_test.assign(split="test")]).to_csv(
        config.ARTIFACTS / "cost_sweep.csv", index=False)

    single = decision.best_single(sweep_val)
    t_star = float(single["threshold"])
    band = decision.best_band(decision.sweep_band(val.score, val.label, costs, config.BASE_RATE))
    t_low, t_high = float(band["t_low"]), float(band["t_high"])

    alone_test = decision.evaluate(test.score, test.label, t_star, t_star, costs, config.BASE_RATE)
    band_test = decision.evaluate(test.score, test.label, t_low, t_high, costs, config.BASE_RATE)
    alts = decision.alternatives(alone_test, band_test, costs, config.BASE_RATE)
    alts.to_csv(config.ARTIFACTS / "alternatives.csv", index=False)
    sens = sensitivity(val.score.to_numpy(), val.label.to_numpy(), costs)
    sens.to_csv(config.ARTIFACTS / "sensitivity.csv", index=False)
    plot_cost_curve(sweep_val, sweep_test, t_star)

    op = {
        "assumptions": {
            "base_rate": config.BASE_RATE, "cost_fn": config.COST_FN, "cost_fp": config.COST_FP,
            "cost_tp": config.COST_TP, "cost_tn": config.COST_TN, "cost_review": config.COST_REVIEW,
            "hourly_wage": config.HOURLY_WAGE, "seconds_per_review": config.SECONDS_PER_REVIEW,
            "human_recall": config.HUMAN_RECALL, "human_false_fail": config.HUMAN_FALSE_FAIL,
            "eggs_per_day": config.EGGS_PER_DAY,
            "productive_hours_per_shift": config.PRODUCTIVE_HOURS_PER_SHIFT,
        },
        "bayes_threshold_if_scores_were_calibrated": config.COST_FP / (config.COST_FP + config.COST_FN),
        "positives_in_split": {"valid": int(val.label.sum()), "test": int(test.label.sum())},
        "negatives_in_split": {"valid": int((val.label == 0).sum()), "test": int((test.label == 0).sum())},
        "single_threshold": {
            "chosen_on": "valid", "threshold": t_star,
            "valid": decision.evaluate(val.score, val.label, t_star, t_star, costs, config.BASE_RATE),
            "test": alone_test,
            "test_cost_95ci": bootstrap_ci(test.score, test.label, t_star, t_star, costs),
        },
        "review_band": {
            "chosen_on": "valid", "t_low": t_low, "t_high": t_high,
            "valid": decision.evaluate(val.score, val.label, t_low, t_high, costs, config.BASE_RATE),
            "test": band_test,
            "test_cost_95ci": bootstrap_ci(test.score, test.label, t_low, t_high, costs),
            "staffing": decision.staffing(band_test["review_share"]),
        },
    }
    save_json(op, "operating_point.json")
    print(json.dumps(op, indent=2, default=float))
    print(alts.to_string(index=False))
    return test, op, costs


def save_examples(scores_test: pd.DataFrame) -> None:
    """A few test images for the dashboard (CC BY 4.0, credited in REPORT.md)."""
    out = config.ARTIFACTS / "examples"
    out.mkdir(exist_ok=True)
    picks = pd.concat([scores_test[scores_test.label == 1].head(2), scores_test[scores_test.label == 0].head(2)])
    folder = config.DATASET_DIR / "test" / "images"
    for name in picks.image:
        shutil.copy(folder / name, out / name)


def main() -> None:
    global POS_IDX
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    config.ARTIFACTS.mkdir(exist_ok=True)

    download_dataset()
    data_yaml, names = write_data_yaml()
    lowered = [n.lower() for n in names]
    POS_IDX = lowered.index(config.POSITIVE_CLASS.lower())
    log(f"Classes: {names}. Positive class: '{names[POS_IDX]}' (index {POS_IDX})")

    describe_dataset(names, POS_IDX)
    model = finetune(data_yaml)
    det = detection_metrics(model, data_yaml, names)
    scores_test, op, costs = operating_point(model, names)
    base = baselines(model, scores_test, op["single_threshold"]["threshold"], names, costs)
    det["baselines"] = base
    save_json(det, "detection_metrics.json")
    save_examples(scores_test)
    save_json({"classes": names, "positive_class": names[POS_IDX], "positive_index": POS_IDX,
               "weights": str(config.WEIGHTS_PATH.relative_to(config.ROOT)),
               "base_model": config.BASE_MODEL, "epochs": config.EPOCHS, "patience": config.PATIENCE,
               "imgsz": config.IMGSZ, "batch": config.BATCH, "seed": config.SEED,
               "device": pick_device(), "torch": torch.__version__},
              "run_info.json")
    print(json.dumps(det, indent=2, default=float))
    log("Done. Commit weights/ and artifacts/.")


POS_IDX = 1

if __name__ == "__main__":
    main()
