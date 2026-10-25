"""
Assignment 2: Gradio dashboard (run with `uv run python main.py app`).

Loads the committed weights in weights/ and the outputs train.py saved in
artifacts/. It never trains anything. Every number below is recomputed from
artifacts/image_scores.csv with the same functions train.py used (decision.py),
so moving a slider here gives exactly what the pipeline would give.
"""

import json

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO

import config
import decision

A = config.ARTIFACTS
RUN = json.loads((A / "run_info.json").read_text())
OP = json.loads((A / "operating_point.json").read_text())
DET = json.loads((A / "detection_metrics.json").read_text())
STATS = json.loads((A / "dataset_stats.json").read_text())
SCORES = pd.read_csv(A / "image_scores.csv")
MODEL = YOLO(str(config.ROOT / RUN["weights"]))
POS_IDX = RUN["positive_index"]

T_STAR = OP["single_threshold"]["threshold"]
T_LOW0, T_HIGH0 = OP["review_band"]["t_low"], OP["review_band"]["t_high"]
EXAMPLES = sorted(str(p) for p in (A / "examples").glob("*")) if (A / "examples").exists() else []


def make_costs(cost_fn, cost_fp, seconds):
    return decision.Costs(fn=cost_fn, fp=cost_fp, review=config.HOURLY_WAGE * seconds / 3600)


# --- 1. Check one egg ------------------------------------------------------------

def check_egg(image, t_low, t_high):
    if image is None:
        return None, "Upload a photo of an egg, or pick one of the examples."
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    res = MODEL.predict(image, conf=max(min(t_low, 0.99), 0.01), imgsz=config.IMGSZ, verbose=False)[0]
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    score = float(conf[cls == POS_IDX].max()) if np.any(cls == POS_IDX) else 0.0
    outcome = decision.OUTCOMES[int(decision.decide([score], t_low, t_high)[0])]
    annotated = res.plot()[:, :, ::-1]  # BGR -> RGB
    verdict = {
        "pass": "### ✅ PASS: pack as class A",
        "review": "### 🟡 REVIEW: send to a person",
        "fail": "### ❌ FAIL: divert to the industrial channel",
    }[outcome]
    detail = (f"{verdict}\n\nHighest *cracked egg* confidence: **{score:.3f}**  \n"
              f"Pass below {t_low:.2f} · review between {t_low:.2f} and {t_high:.2f} · fail at or above {t_high:.2f}.  \n"
              f"Boxes shown: every detection with confidence ≥ {t_low:.2f}.")
    return annotated, detail


# --- 2. Decision, cost and review load -----------------------------------------

def heatmap(df, title, fmt):
    fig = go.Figure(go.Heatmap(
        z=df.to_numpy(), x=list(df.columns), y=list(df.index), colorscale="Blues", showscale=False,
        text=[[fmt.format(v) for v in row] for row in df.to_numpy()], texttemplate="%{text}",
        textfont={"size": 16}))
    fig.update_layout(title=title, height=260, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="decision", yaxis_title="truth", yaxis_autorange="reversed")
    return fig


def cost_curve(split_df, costs, base_rate, t_low, t_high):
    sweep = decision.sweep_single(split_df.score, split_df.label, costs, base_rate)
    best = decision.best_single(sweep)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep.threshold, y=sweep.cost_per_1000, mode="lines",
                             name="model decides alone"))
    fig.add_vline(x=T_STAR, line_dash="dash", line_color="black",
                  annotation_text=f"chosen in training: {T_STAR:.2f}", annotation_position="top right")
    fig.add_vline(x=float(best.threshold), line_dash="dot", line_color="green",
                  annotation_text=f"cheapest now: {best.threshold:.2f}", annotation_position="bottom right")
    fig.add_vrect(x0=t_low, x1=t_high, fillcolor="orange", opacity=0.15, line_width=0,
                  annotation_text="current review band", annotation_position="top left")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="confidence threshold for 'cracked egg'",
                      yaxis_title="expected cost per 1,000 eggs (EUR)",
                      title="Expected cost against threshold")
    return fig


def update(t_low, t_high, split, base_rate, cost_fn, cost_fp, seconds, eggs_per_day):
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    costs = make_costs(cost_fn, cost_fp, seconds)
    df = SCORES[SCORES.split == split]
    band = decision.evaluate(df.score, df.label, t_low, t_high, costs, base_rate)
    rates = {k: band[k] for k in band if k.startswith(("pos_", "neg_"))}
    raw = decision.outcome_counts(df.score, df.label, t_low, t_high)
    per_1000 = decision.expected_matrix_per_1000(rates, base_rate)
    staff = decision.staffing(band["review_share"], eggs_per_day, seconds)

    alone = decision.evaluate(df.score, df.label, T_STAR, T_STAR, costs, base_rate)
    alts = decision.alternatives(alone, band, costs, base_rate)
    alts["cost_per_day_eur"] = (alts["cost_per_1000_eggs"] * eggs_per_day / 1000).round(2)
    alts["people_needed"] = alts["share_to_person"].map(
        lambda s: round(decision.staffing(s, eggs_per_day, seconds)["people_needed"], 2))
    alts = alts.rename(columns={"option": "Option", "cost_per_1000_eggs": "€ / 1,000 eggs",
                                "cost_per_day_eur": "€ / day", "share_to_person": "share to a person",
                                "people_needed": "people needed",
                                "cracked_eggs_shipped_per_1000": "cracked shipped / 1,000"})

    kpis = (f"## €{band['cost_per_1000']:.2f} per 1,000 eggs · "
            f"{band['review_share']:.1%} sent to review\n"
            f"That is **{staff['eggs_reviewed_per_day']:,.0f} eggs/day** to check, "
            f"**{staff['person_hours_per_day']:.1f} person-hours**, about "
            f"**{staff['people_needed']:.2f} people** at {config.PRODUCTIVE_HOURS_PER_SHIFT:.0f} productive h/shift.  \n"
            f"Measured on the **{split}** split ({int(df.label.sum())} cracked, {int((df.label == 0).sum())} sound eggs); "
            f"costed at a production base rate of **{base_rate:.1%}** cracked.")
    return (
        kpis,
        heatmap(raw, f"Counts on the {split} split", "{:.0f}"),
        heatmap(per_1000, f"Expected out of 1,000 production eggs (base rate {base_rate:.0%})", "{:.1f}"),
        alts,
        cost_curve(df, costs, base_rate, t_low, t_high),
    )


# --- 3. Per-class results ----------------------------------------------------------

def per_class_table():
    rows = [{"class": "ALL (mean)", **DET["overall"], "AP50-95": DET["overall"]["mAP50-95"]}]
    rows[0].pop("mAP50-95", None)
    rows += DET["per_class"]
    return pd.DataFrame(rows).round(3)


def baselines_table():
    lvl = DET["baselines"]["image_level_decision_on_test"]
    df = pd.DataFrame(lvl).T.reset_index().rename(columns={"index": "model / rule"})
    return df[["model / rule", "accuracy", "recall_cracked", "false_fail_rate", "TP", "FP", "FN", "TN",
               "cost_per_1000"]].round(3)


def detection_ap_table():
    return pd.DataFrame([
        {"model": "finetuned yolo11s", "AP50": DET["overall"]["AP50"], "mAP50-95": DET["overall"]["mAP50-95"]},
        {"model": "COCO yolo11s zero-shot", "AP50": 0.0, "mAP50-95": 0.0},
        {"model": "predict nothing", "AP50": 0.0, "mAP50-95": 0.0},
    ]).round(3)


def confusion_figure():
    cm = DET["confusion_matrix"]
    df = pd.DataFrame(cm["rows_predicted_cols_true"], index=[f"pred: {l}" for l in cm["labels"]],
                      columns=[f"true: {l}" for l in cm["labels"]])
    fig = heatmap(df, f"Box-level confusion matrix on test (confidence {cm.get('confidence', 0.25)})", "{:.0f}")
    fig.update_layout(height=340, xaxis_title="", yaxis_title="")
    return fig


def dataset_table():
    inst = pd.DataFrame(STATS["instances_per_class_and_split"]).T
    inst["images"] = pd.Series(STATS["images_per_split"])
    return inst.reset_index().rename(columns={"index": "split"})


# --- Layout -------------------------------------------------------------------------

with gr.Blocks(title="Cracked egg grading") as demo:
    gr.Markdown(
        "# 🥚 Cracked-egg grading: where should the model stop and a person start?\n"
        f"Finetuned **{RUN['base_model']}** on Roboflow `{config.RF_WORKSPACE}/{config.RF_PROJECT}` "
        f"v{config.RF_VERSION}. Positive class: **{RUN['positive_class']}**. "
        f"Thresholds were chosen on the validation split; the test split is the honest estimate.")

    with gr.Row():
        t_low = gr.Slider(0, 1, value=T_LOW0, step=0.01, label="Lower threshold: below this, PASS")
        t_high = gr.Slider(0, 1, value=T_HIGH0, step=0.01, label="Upper threshold: at or above this, FAIL")
        split = gr.Radio(["test", "valid"], value="test", label="Measure on split")
    with gr.Accordion("Assumptions (defaults argued in REPORT.md)", open=False):
        with gr.Row():
            base_rate = gr.Slider(0.005, 0.30, value=config.BASE_RATE, step=0.005, label="Base rate (share cracked)")
            cost_fn = gr.Number(value=config.COST_FN, label="€ per missed crack (FN)")
            cost_fp = gr.Number(value=config.COST_FP, label="€ per sound egg failed (FP)")
            seconds = gr.Number(value=config.SECONDS_PER_REVIEW, label="Seconds per human review")
            eggs_day = gr.Number(value=config.EGGS_PER_DAY, label="Eggs per day", precision=0)
        reset = gr.Button("Reset thresholds to the chosen operating point")

    with gr.Tabs():
        with gr.Tab("1 · Check an egg"):
            with gr.Row():
                img_in = gr.Image(type="numpy", label="Egg photo")
                img_out = gr.Image(label="Model output")
            verdict = gr.Markdown()
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=img_in, label="Test-split examples")
        with gr.Tab("2 · Decision, cost and review load"):
            kpis = gr.Markdown()
            with gr.Row():
                cm_raw = gr.Plot(label="Raw counts")
                cm_1000 = gr.Plot(label="Per 1,000 eggs")
            gr.Markdown("### Against the alternatives")
            alts = gr.Dataframe(interactive=False)
        with gr.Tab("3 · Cost curve"):
            curve = gr.Plot()
            gr.Markdown(f"Black dashed line: threshold chosen by train.py on validation ({T_STAR:.2f}). "
                        "Green dotted: cheapest threshold under the current assumptions and split. "
                        "Orange band: the current review band.")
        with gr.Tab("4 · Per-class results"):
            gr.Markdown("### Detection metrics on test, per class (not just the average)")
            gr.Dataframe(per_class_table(), interactive=False)
            gr.Markdown("### Against the baselines: detection")
            gr.Dataframe(detection_ap_table(), interactive=False)
            gr.Markdown("### Against the baselines: the pass/fail decision on test eggs (model alone)")
            gr.Dataframe(baselines_table(), interactive=False)
            gr.Plot(confusion_figure())
            gr.Markdown("### Dataset composition")
            gr.Dataframe(dataset_table(), interactive=False)
            if (A / "dataset_overview.png").exists():
                gr.Image(str(A / "dataset_overview.png"), label="Instances and box sizes", interactive=False)

    controls = [t_low, t_high, split, base_rate, cost_fn, cost_fp, seconds, eggs_day]
    outputs = [kpis, cm_raw, cm_1000, alts, curve]
    for c in controls:
        c.change(update, controls, outputs)
    for c in [img_in, t_low, t_high]:
        c.change(check_egg, [img_in, t_low, t_high], [img_out, verdict])
    reset.click(lambda: (T_LOW0, T_HIGH0), None, [t_low, t_high])
    demo.load(update, controls, outputs)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
