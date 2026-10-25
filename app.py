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


# --- Visual system ------------------------------------------------------------------
# One colour per decision, used everywhere: pass = green, review = yolk, fail = terracotta.

INK, MUTED, LINE = "#1C1A17", "#8A8276", "rgba(138,130,118,0.18)"
PASS, REVIEW, FAIL = "#2F8F6A", "#E39B2D", "#C4553D"
HEAT = [[0, "#FBF6EC"], [0.35, "#F6DDAE"], [0.7, "#EDB45E"], [1, "#E0952C"]]
FONT = "Inter, ui-sans-serif, system-ui, sans-serif"


def style(fig, height=360, title=None):
    fig.update_layout(
        height=height, title=dict(text=title, x=0.01, y=0.97, yanchor="top", font=dict(size=15, color=MUTED)) if title else None,
        font=dict(family=FONT, size=13, color=MUTED),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=44 if title else 16, b=8),
        hoverlabel=dict(font_family=FONT, bgcolor="#1C1A17", font_color="#FAF8F3", bordercolor="#1C1A17"),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=12)),
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, ticks="")
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, ticks="")
    return fig


def make_costs(cost_fn, cost_fp, seconds):
    return decision.Costs(fn=cost_fn, fp=cost_fp, review=config.HOURLY_WAGE * seconds / 3600)


def eur(x, d=2):
    return f"€{x:,.{d}f}"


def kpi(label, value, note="", tone=""):
    return (f'<div class="eg-kpi {tone}"><div class="eg-kpi-label">{label}</div>'
            f'<div class="eg-kpi-value">{value}</div><div class="eg-kpi-note">{note}</div></div>')


# --- 1. Check one egg ------------------------------------------------------------

def gauge(score, t_low, t_high):
    lo, hi = 100 * t_low, 100 * t_high
    return f"""
    <div class="eg-gauge">
      <div class="eg-gauge-track">
        <span style="left:0;width:{lo}%;background:{PASS}"></span>
        <span style="left:{lo}%;width:{hi - lo}%;background:{REVIEW}"></span>
        <span style="left:{hi}%;width:{100 - hi}%;background:{FAIL}"></span>
        <i style="left:{100 * score}%"></i>
      </div>
      <div class="eg-gauge-axis"><span>0</span><span>pass &lt; {t_low:.2f}</span>
        <span>review</span><span>fail ≥ {t_high:.2f}</span><span>1</span></div>
    </div>"""


def check_egg(image, t_low, t_high):
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    if image is None:
        return None, ('<div class="eg-verdict eg-empty"><div class="eg-verdict-label">No egg yet</div>'
                      '<div class="eg-verdict-sub">Upload a photo or pick one of the test-split examples below.</div></div>')
    res = MODEL.predict(image, conf=max(min(t_low, 0.99), 0.01), imgsz=config.IMGSZ, verbose=False)[0]
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    score = float(conf[cls == POS_IDX].max()) if np.any(cls == POS_IDX) else 0.0
    outcome = decision.OUTCOMES[int(decision.decide([score], t_low, t_high)[0])]
    annotated = res.plot()[:, :, ::-1]  # BGR -> RGB
    text = {"pass": ("Pass", "Pack as class A."),
            "review": ("Review", "Send to a person: the model is not sure enough either way."),
            "fail": ("Fail", "Divert to the industrial channel.")}[outcome]
    html = f"""
    <div class="eg-verdict eg-{outcome}">
      <div class="eg-verdict-top"><span class="eg-dot"></span><span class="eg-verdict-label">{text[0]}</span>
        <span class="eg-verdict-score">cracked score <b>{score:.3f}</b></span></div>
      <div class="eg-verdict-sub">{text[1]} Boxes shown: every detection with confidence ≥ {t_low:.2f}.</div>
      {gauge(score, t_low, t_high)}
    </div>"""
    return annotated, html


# --- 2. Decision, cost and review load -----------------------------------------

def heatmap(df, fmt, title):
    z = df.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=[c.capitalize() for c in df.columns], y=list(df.index), colorscale=HEAT, showscale=False,
        xgap=4, ygap=4, text=[[fmt.format(v) for v in row] for row in z], texttemplate="%{text}",
        textfont=dict(size=20, color=INK, family=FONT), hovertemplate="%{y} → %{x}: %{text}<extra></extra>"))
    style(fig, 270, title)
    fig.update_layout(margin=dict(l=12, r=12, t=78, b=8))
    fig.update_xaxes(side="top", showgrid=False, tickfont=dict(size=12))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=12))
    return fig


def routing_bar(rates, base_rate):
    shares = [base_rate * rates[f"pos_{k}"] + (1 - base_rate) * rates[f"neg_{k}"] for k in decision.OUTCOMES]
    colours = [PASS, REVIEW, FAIL]
    segs = "".join(f'<span style="width:{100 * s:.2f}%;background:{c}"></span>' for s, c in zip(shares, colours))
    legend = "".join(f'<div><i style="background:{c}"></i>{n}<b>{s:.1%}</b></div>'
                     for n, s, c in zip(["Passed by the model", "Sent to a person", "Failed by the model"], shares, colours))
    return f'<div class="eg-route"><div class="eg-route-bar">{segs}</div><div class="eg-route-legend">{legend}</div></div>'


def alternatives_chart(alts):
    order = alts.sort_values("cost_per_1000_eggs", ascending=True)
    colours = [REVIEW if o == "Model + human review band" else "#CFC6B6" for o in order.option]
    fig = go.Figure(go.Bar(
        x=order.cost_per_1000_eggs, y=order.option, orientation="h", marker_color=colours,
        text=[eur(v) for v in order.cost_per_1000_eggs], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK, size=13), hovertemplate="%{y}: €%{x:.2f} per 1,000<extra></extra>"))
    style(fig, 250, "Expected cost per 1,000 eggs, by strategy")
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(autorange="reversed", tickfont=dict(color=INK, size=13))
    fig.update_layout(margin=dict(l=8, r=60, t=44, b=8), bargap=0.45)
    return fig


def cost_curve(split_df, costs, base_rate, t_low, t_high):
    sweep = decision.sweep_single(split_df.score, split_df.label, costs, base_rate)
    best = decision.best_single(sweep)
    fig = go.Figure()
    fig.add_vrect(x0=t_low, x1=t_high, fillcolor=REVIEW, opacity=0.10, line_width=0)
    fig.add_trace(go.Scatter(x=sweep.threshold, y=sweep.cost_per_1000, mode="lines", name="Model decides alone",
                             line=dict(color=INK, width=2.5), hovertemplate="t = %{x:.2f}<br>€%{y:.2f} per 1,000<extra></extra>"))
    fig.add_trace(go.Scatter(x=[T_STAR], y=[sweep.loc[np.isclose(sweep.threshold, T_STAR), "cost_per_1000"].iloc[0]],
                             mode="markers", name=f"Chosen in training ({T_STAR:.2f})",
                             marker=dict(size=13, color=REVIEW, line=dict(color=INK, width=2))))
    fig.add_trace(go.Scatter(x=[best.threshold], y=[best.cost_per_1000], mode="markers",
                             name=f"Cheapest under current inputs ({best.threshold:.2f})",
                             marker=dict(size=11, color=PASS, symbol="diamond")))
    style(fig, 440)
    fig.update_xaxes(title="Confidence threshold for “cracked egg”", range=[0, 1])
    fig.update_yaxes(title="Expected € per 1,000 eggs", rangemode="tozero")
    return fig


def update(t_low, t_high, split, base_rate, cost_fn, cost_fp, seconds, eggs_per_day):
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    costs = make_costs(cost_fn, cost_fp, seconds)
    df = SCORES[SCORES.split == split]
    band = decision.evaluate(df.score, df.label, t_low, t_high, costs, base_rate)
    rates = {k: band[k] for k in band if k.startswith(("pos_", "neg_"))}
    raw = decision.outcome_counts(df.score, df.label, t_low, t_high)
    raw.index = ["Cracked", "Sound"]
    per_1000 = decision.expected_matrix_per_1000(rates, base_rate)
    per_1000.index = ["Cracked", "Sound"]
    staff = decision.staffing(band["review_share"], eggs_per_day, seconds)

    alone = decision.evaluate(df.score, df.label, T_STAR, T_STAR, costs, base_rate)
    alts = decision.alternatives(alone, band, costs, base_rate)
    alts["cost_per_day_eur"] = (alts["cost_per_1000_eggs"] * eggs_per_day / 1000).round(2)
    alts["people_needed"] = alts["share_to_person"].map(
        lambda s: round(decision.staffing(s, eggs_per_day, seconds)["people_needed"], 2))
    everyone = alts.loc[alts.option == "A person checks every egg", "cost_per_1000_eggs"].iloc[0]
    shipped = alts.loc[alts.option == "Model + human review band", "cracked_eggs_shipped_per_1000"].iloc[0]
    delta = band["cost_per_1000"] - everyone
    tone = "eg-good" if delta < 0 else "eg-bad"

    kpis = ('<div class="eg-kpis">'
            + kpi("Expected cost", eur(band["cost_per_1000"]), "per 1,000 eggs")
            + kpi("vs. a person checking all", f"{'+' if delta >= 0 else '−'}{eur(abs(delta))}",
                  "per 1,000 eggs" + (" · cheaper" if delta < 0 else " · dearer"), tone)
            + kpi("Sent to review", f"{band['review_share']:.1%}",
                  f"{staff['eggs_reviewed_per_day']:,.0f} eggs a day")
            + kpi("People needed", f"{staff['people_needed']:.2f}",
                  f"{staff['person_hours_per_day']:.1f} person-hours a day")
            + kpi("Cracked eggs shipped", f"{shipped:.1f}", "per 1,000 eggs")
            + "</div>"
            + f'<p class="eg-caption">Measured on the <b>{split}</b> split ({int(df.label.sum())} cracked, '
              f'{int((df.label == 0).sum())} sound eggs), re-weighted to a production base rate of '
              f'<b>{base_rate:.1%}</b> cracked. {config.PRODUCTIVE_HOURS_PER_SHIFT:.0f} productive hours per shift.</p>')

    table = alts.rename(columns={"option": "Strategy", "cost_per_1000_eggs": "€ / 1,000 eggs",
                                 "cost_per_day_eur": "€ / day", "share_to_person": "Share to a person",
                                 "people_needed": "People needed",
                                 "cracked_eggs_shipped_per_1000": "Cracked shipped / 1,000"})
    return (
        kpis,
        routing_bar(rates, base_rate),
        heatmap(raw, "{:.0f}", f"Counts on the {split} split · truth × decision"),
        heatmap(per_1000, "{:.1f}", f"Expected per 1,000 production eggs · base rate {base_rate:.0%}"),
        alternatives_chart(alts),
        table,
        cost_curve(df, costs, base_rate, t_low, t_high),
    )


# --- 3. Per-class results ----------------------------------------------------------

def class_cards():
    cards = []
    for row in DET["per_class"]:
        positive = row["class"] == RUN["positive_class"]
        tag = '<span class="eg-tag">positive class</span>' if positive else ""
        stats = "".join(f'<div><span>{k}</span><b>{row[k]:.3f}</b></div>' for k in ["precision", "recall", "AP50", "AP50-95"])
        cards.append(f'<div class="eg-class {"eg-pos" if positive else ""}"><div class="eg-class-head">'
                     f'{row["class"]}{tag}</div><div class="eg-class-stats">{stats}</div></div>')
    o = DET["overall"]
    overall = "".join(f'<div><span>{k}</span><b>{o[k]:.3f}</b></div>' for k in ["precision", "recall", "AP50", "mAP50-95"])
    cards.append(f'<div class="eg-class eg-overall"><div class="eg-class-head">All classes (mean)</div>'
                 f'<div class="eg-class-stats">{overall}</div></div>')
    return '<div class="eg-classes">' + "".join(cards) + "</div>"


def baselines_chart():
    lvl = DET["baselines"]["image_level_decision_on_test"]
    names = list(lvl)
    vals = [lvl[n]["cost_per_1000"] for n in names]
    pretty = ["Finetuned model", "COCO zero-shot", "Predict nothing", "Majority class"]
    fig = go.Figure(go.Bar(
        x=pretty, y=vals, marker_color=[REVIEW, "#CFC6B6", "#CFC6B6", "#CFC6B6"],
        text=[eur(v) for v in vals], textposition="outside", cliponaxis=False, textfont=dict(color=INK),
        hovertemplate="%{x}: €%{y:.2f} per 1,000<extra></extra>"))
    style(fig, 300, "Model alone vs. baselines · € per 1,000 eggs on test")
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_xaxes(tickfont=dict(color=INK), showgrid=False)
    fig.update_layout(bargap=0.5)
    return fig


def baselines_table():
    lvl = DET["baselines"]["image_level_decision_on_test"]
    df = pd.DataFrame(lvl).T.reset_index().rename(columns={"index": "Model / rule"})
    return df[["Model / rule", "accuracy", "recall_cracked", "false_fail_rate", "TP", "FP", "FN", "TN",
               "cost_per_1000"]].round(3)


def detection_ap_table():
    return pd.DataFrame([
        {"Model": "Finetuned yolo11s", "AP50": DET["overall"]["AP50"], "mAP50-95": DET["overall"]["mAP50-95"]},
        {"Model": "COCO yolo11s zero-shot", "AP50": 0.0, "mAP50-95": 0.0},
        {"Model": "Predict nothing", "AP50": 0.0, "mAP50-95": 0.0},
    ]).round(3)


def confusion_figure():
    cm = DET["confusion_matrix"]
    labels = [l.capitalize() for l in cm["labels"]]
    df = pd.DataFrame(cm["rows_predicted_cols_true"], index=[f"Pred · {l}" for l in labels],
                      columns=[f"True · {l}" for l in labels])
    fig = heatmap(df, "{:.0f}", f"Box-level confusion matrix on test · confidence {cm.get('confidence', 0.25)}")
    fig.update_layout(height=340)
    return fig


def dataset_charts():
    inst = pd.DataFrame(STATS["instances_per_class_and_split"]).T.loc[["train", "valid", "test"]]
    bars = go.Figure()
    for cls, colour in [(c, REVIEW if c == RUN["positive_class"] else "#CFC6B6") for c in inst.columns]:
        bars.add_trace(go.Bar(x=[s.capitalize() for s in inst.index], y=inst[cls], name=cls.capitalize(),
                              marker_color=colour, text=inst[cls], textposition="outside", cliponaxis=False,
                              textfont=dict(color=INK)))
    style(bars, 320, "Instances per class and split")
    bars.update_layout(barmode="group", bargap=0.35)
    bars.update_yaxes(showticklabels=False, showgrid=False)
    bars.update_xaxes(showgrid=False)
    sizes = go.Figure()
    box = pd.read_csv(A / "box_sizes.csv") if (A / "box_sizes.csv").exists() else pd.DataFrame()
    for cls, colour in [(c, REVIEW if c == RUN["positive_class"] else "#B9AE9B") for c in sorted(box.get("class", pd.Series()).unique())]:
        sizes.add_trace(go.Histogram(x=np.sqrt(box.loc[box["class"] == cls, "area_frac"]), name=cls.capitalize(),
                                     marker_color=colour, opacity=0.7, nbinsx=30, histnorm="percent"))
    style(sizes, 320, "Object size · √(box area ÷ image area)")
    sizes.update_layout(barmode="overlay", legend=dict(orientation="v", x=1, xanchor="right", y=0.98, yanchor="top"))
    sizes.update_xaxes(title="Relative side length of the egg box")
    sizes.update_yaxes(title="% of boxes in class", ticksuffix="%")
    return bars, sizes


def dataset_table():
    inst = pd.DataFrame(STATS["instances_per_class_and_split"]).T
    inst["images"] = pd.Series(STATS["images_per_split"])
    return inst.reset_index().rename(columns={"index": "Split"})


# --- Layout -------------------------------------------------------------------------

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber, neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#F7F4EE", body_background_fill_dark="#14130F",
    body_text_color=INK, body_text_color_dark="#F1ECE3",
    block_background_fill="#FFFFFF", block_background_fill_dark="#1D1B17",
    block_border_color="#ECE6DA", block_border_color_dark="#2D2A24",
    block_shadow="none", block_label_text_color=MUTED,
    slider_color=REVIEW, slider_color_dark=REVIEW,
    button_primary_background_fill=INK, button_primary_background_fill_hover="#3A362F",
    button_primary_text_color="#FAF8F3",
    button_primary_background_fill_dark="#F1ECE3", button_primary_text_color_dark=INK,
)

HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&display=swap" rel="stylesheet">')

CSS = f"""
:root {{ --eg-ink:{INK}; --eg-muted:{MUTED}; --eg-line:#ECE6DA; --eg-card:#FFFFFF; --eg-soft:#FBF8F2;
        --eg-pass:{PASS}; --eg-review:{REVIEW}; --eg-fail:{FAIL}; }}
.dark {{ --eg-ink:#F1ECE3; --eg-muted:#A79F92; --eg-line:#2D2A24; --eg-card:#1D1B17; --eg-soft:#23211C; }}
.gradio-container {{ max-width: 1240px !important; margin: 0 auto !important; }}
.eg-hero {{ padding: 28px 4px 8px; }}
.eg-eyebrow {{ font-size: 12px; letter-spacing: .14em; text-transform: uppercase; color: var(--eg-muted); }}
.eg-eyebrow b {{ color: var(--eg-review); font-weight: 600; }}
.eg-title {{ font-family: 'Fraunces', Georgia, serif; font-weight: 600; font-size: clamp(28px, 4vw, 44px);
             line-height: 1.08; letter-spacing: -.01em; color: var(--eg-ink); margin: 10px 0 12px; }}
.eg-title em {{ font-style: italic; font-weight: 400; color: var(--eg-review); }}
.eg-lede {{ color: var(--eg-muted); max-width: 760px; font-size: 15px; line-height: 1.55; margin: 0; }}
.eg-chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }}
.eg-chip {{ border: 1px solid var(--eg-line); background: var(--eg-card); border-radius: 999px;
            padding: 5px 12px; font-size: 12.5px; color: var(--eg-muted); }}
.eg-chip b {{ color: var(--eg-ink); font-weight: 600; }}
.eg-controls {{ border: 1px solid var(--eg-line) !important; border-radius: 16px !important; }}
.eg-kpis {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }}
@media (max-width: 900px) {{ .eg-kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.eg-kpi {{ background: var(--eg-card); border: 1px solid var(--eg-line); border-radius: 14px; padding: 16px 18px; }}
.eg-kpi-label {{ font-size: 12px; color: var(--eg-muted); text-transform: uppercase; letter-spacing: .08em; }}
.eg-kpi-value {{ font-family: 'Fraunces', Georgia, serif; font-size: 32px; font-weight: 600; color: var(--eg-ink);
                 margin: 6px 0 2px; font-variant-numeric: tabular-nums; letter-spacing: -.01em; }}
.eg-kpi-note {{ font-size: 12.5px; color: var(--eg-muted); }}
.eg-kpi.eg-good .eg-kpi-value {{ color: var(--eg-pass); }}
.eg-kpi.eg-bad .eg-kpi-value {{ color: var(--eg-fail); }}
.eg-caption {{ color: var(--eg-muted); font-size: 13px; margin: 12px 2px 0; }}
.eg-section {{ font-family: 'Fraunces', Georgia, serif; font-size: 21px; font-weight: 600; color: var(--eg-ink);
               margin: 22px 2px 2px; }}
.eg-section-sub {{ color: var(--eg-muted); font-size: 13.5px; margin: 2px 2px 10px; }}
.eg-route-bar {{ display: flex; height: 18px; border-radius: 999px; overflow: hidden; background: var(--eg-soft); }}
.eg-route-bar span {{ display: block; height: 100%; transition: width .35s ease; }}
.eg-route-legend {{ display: flex; flex-wrap: wrap; gap: 22px; margin-top: 10px; font-size: 13px; color: var(--eg-muted); }}
.eg-route-legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 7px; }}
.eg-route-legend b {{ color: var(--eg-ink); margin-left: 6px; font-variant-numeric: tabular-nums; }}
.eg-verdict {{ border: 1px solid var(--eg-line); border-left: 6px solid var(--eg-muted); background: var(--eg-card);
               border-radius: 14px; padding: 18px 20px; }}
.eg-verdict.eg-pass {{ border-left-color: var(--eg-pass); }} .eg-verdict.eg-review {{ border-left-color: var(--eg-review); }}
.eg-verdict.eg-fail {{ border-left-color: var(--eg-fail); }}
.eg-verdict-top {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
.eg-dot {{ width: 12px; height: 12px; border-radius: 50%; background: var(--eg-muted); }}
.eg-pass .eg-dot {{ background: var(--eg-pass); }} .eg-review .eg-dot {{ background: var(--eg-review); }}
.eg-fail .eg-dot {{ background: var(--eg-fail); }}
.eg-verdict-label {{ font-family: 'Fraunces', Georgia, serif; font-size: 30px; font-weight: 600; color: var(--eg-ink); }}
.eg-verdict-score {{ margin-left: auto; color: var(--eg-muted); font-size: 13px; }}
.eg-verdict-score b {{ color: var(--eg-ink); font-size: 16px; font-variant-numeric: tabular-nums; }}
.eg-verdict-sub {{ color: var(--eg-muted); font-size: 13.5px; margin: 6px 0 14px; }}
.eg-gauge-track {{ position: relative; height: 10px; border-radius: 999px; overflow: visible; background: var(--eg-soft); }}
.eg-gauge-track span {{ position: absolute; top: 0; height: 100%; opacity: .85; }}
.eg-gauge-track span:first-child {{ border-radius: 999px 0 0 999px; }}
.eg-gauge-track span:nth-child(3) {{ border-radius: 0 999px 999px 0; }}
.eg-gauge-track i {{ position: absolute; top: -5px; width: 4px; height: 20px; margin-left: -2px; border-radius: 2px;
                     background: var(--eg-ink); box-shadow: 0 0 0 3px var(--eg-card); }}
.eg-gauge-axis {{ display: flex; justify-content: space-between; font-size: 11.5px; color: var(--eg-muted); margin-top: 8px; }}
.eg-classes {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
@media (max-width: 900px) {{ .eg-classes {{ grid-template-columns: 1fr; }} }}
.eg-class {{ background: var(--eg-card); border: 1px solid var(--eg-line); border-radius: 14px; padding: 16px 18px; }}
.eg-class.eg-pos {{ border-color: var(--eg-review); box-shadow: inset 0 3px 0 var(--eg-review); }}
.eg-class.eg-overall {{ background: var(--eg-soft); }}
.eg-class-head {{ font-family: 'Fraunces', Georgia, serif; font-size: 19px; font-weight: 600; color: var(--eg-ink);
                  display: flex; align-items: center; gap: 10px; margin-bottom: 12px; text-transform: capitalize; }}
.eg-tag {{ font-family: Inter, sans-serif; font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
           background: var(--eg-review); color: #1C1A17; padding: 2px 8px; border-radius: 999px; }}
.eg-class-stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px 14px; }}
.eg-class-stats span {{ display: block; font-size: 11.5px; color: var(--eg-muted); text-transform: uppercase; letter-spacing: .06em; }}
.eg-class-stats b {{ font-size: 22px; color: var(--eg-ink); font-variant-numeric: tabular-nums; font-weight: 600; }}
.eg-empty .eg-verdict-label {{ color: var(--eg-muted); }}
.tab-nav button, button[role="tab"] {{ font-size: 14px !important; }}
footer {{ opacity: .6; }}
.dark .js-plotly-plot text.bartext, .dark .js-plotly-plot .ytick text, .dark .js-plotly-plot .xtick text,
.dark .js-plotly-plot .legendtext {{ fill: var(--eg-ink) !important; }}
"""

HERO = f"""
<div class="eg-hero">
  <div class="eg-eyebrow"><b>●</b>&nbsp; Egg grading · decision dashboard</div>
  <div class="eg-title">Where should the model stop<br>and a <em>person</em> start?</div>
  <p class="eg-lede">Every egg gets a “cracked” score from the finetuned detector. Below the lower threshold it is packed,
  above the upper one it is diverted, and in between a person looks at it. Move the thresholds and every number
  on this page is recomputed from the pipeline's saved scores. Nothing is retrained.</p>
  <div class="eg-chips">
    <span class="eg-chip">Model <b>{RUN['base_model'].replace('.pt', '')} · finetuned</b></span>
    <span class="eg-chip">Data <b>{config.RF_WORKSPACE}/{config.RF_PROJECT} v{config.RF_VERSION}</b></span>
    <span class="eg-chip">Positive class <b>{RUN['positive_class']}</b></span>
    <span class="eg-chip">Chosen band <b>{T_LOW0:.2f} – {T_HIGH0:.2f}</b> on validation</span>
  </div>
</div>"""

with gr.Blocks(title="Cracked-egg grading") as demo:
    gr.HTML(HERO)

    with gr.Group(elem_classes="eg-controls"):
        with gr.Row():
            t_low = gr.Slider(0, 1, value=T_LOW0, step=0.01, label="Lower threshold · below it, PASS")
            t_high = gr.Slider(0, 1, value=T_HIGH0, step=0.01, label="Upper threshold · at or above it, FAIL")
            split = gr.Radio(["test", "valid"], value="test", label="Measure on split", scale=0)
        with gr.Accordion("Assumptions · defaults argued in REPORT.md", open=False):
            with gr.Row():
                base_rate = gr.Slider(0.005, 0.30, value=config.BASE_RATE, step=0.005, label="Base rate · share cracked")
                cost_fn = gr.Number(value=config.COST_FN, label="€ per missed crack (FN)")
                cost_fp = gr.Number(value=config.COST_FP, label="€ per sound egg failed (FP)")
                seconds = gr.Number(value=config.SECONDS_PER_REVIEW, label="Seconds per review")
                eggs_day = gr.Number(value=config.EGGS_PER_DAY, label="Eggs per day", precision=0)
            reset = gr.Button("Reset to the chosen operating point", variant="primary", size="sm")

    with gr.Tabs():
        with gr.Tab("Decision & cost"):
            kpis = gr.HTML()
            gr.HTML('<div class="eg-section">Where production eggs go</div>'
                    '<div class="eg-section-sub">Share of 1,000 real eggs at the current thresholds and base rate.</div>')
            route = gr.HTML()
            with gr.Row():
                cm_raw = gr.Plot(show_label=False)
                cm_1000 = gr.Plot(show_label=False)
            gr.HTML('<div class="eg-section">Against the alternatives</div>'
                    '<div class="eg-section-sub">Doing nothing, a person checking every egg, the model alone, '
                    'and the model with a review band. Highlighted: the design on the sliders.</div>')
            alts_fig = gr.Plot(show_label=False)
            with gr.Accordion("Full table", open=False):
                alts = gr.Dataframe(interactive=False)

        with gr.Tab("Check an egg"):
            with gr.Row(equal_height=True):
                img_in = gr.Image(type="numpy", label="Egg photo", height=380)
                img_out = gr.Image(label="What the model sees", height=380, interactive=False)
            verdict = gr.HTML()
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=img_in, label="Examples from the test split")

        with gr.Tab("Cost curve"):
            gr.HTML('<div class="eg-section">Expected cost against threshold</div>'
                    '<div class="eg-section-sub">Single threshold, model decides alone. Yolk dot: threshold chosen '
                    'by train.py on validation. Green diamond: cheapest under the current inputs. '
                    'Shaded: the current review band.</div>')
            curve = gr.Plot(show_label=False)
            if (A / "band_sensitivity.csv").exists():
                gr.HTML('<div class="eg-section">What would change the answer</div>'
                        '<div class="eg-section-sub">Band re-chosen on validation under each assumption, '
                        'costed on test (€ per 1,000 eggs). Produced by train.py.</div>')
                sens = pd.read_csv(A / "band_sensitivity.csv").rename(columns={
                    "seconds_per_review": "Sec / review", "base_rate": "Base rate", "cost_fn": "€ missed crack",
                    "t_low": "Band low", "t_high": "Band high", "band_cost_test": "Band €",
                    "band_review_share_test": "Share reviewed", "alone_threshold": "Alone t",
                    "alone_cost_test": "Alone €", "person_checks_all": "Person-all €",
                    "do_nothing": "Nothing €", "cheapest": "Cheapest"})
                gr.Dataframe(sens, interactive=False, max_height=420)

        with gr.Tab("Per-class results"):
            gr.HTML('<div class="eg-section">Detection on the test split, per class</div>'
                    '<div class="eg-section-sub">Precision and recall at the confidence that maximises mean F1. '
                    'The average hides that the two classes fail differently.</div>' + class_cards())
            with gr.Row():
                gr.Plot(confusion_figure(), show_label=False)
                gr.Plot(baselines_chart(), show_label=False)
            with gr.Accordion("Baseline tables", open=False):
                gr.Dataframe(detection_ap_table(), interactive=False, label="Detection: AP")
                gr.Dataframe(baselines_table(), interactive=False, label="Pass/fail decision on test eggs")
            gr.HTML('<div class="eg-section">Dataset composition</div>'
                    '<div class="eg-section-sub">Cracked eggs outnumber sound ones roughly 2.6 to 1 in every split, '
                    'the opposite of a real packing line. Costs are re-weighted to the production base rate for that reason.</div>')
            ds_bars, ds_sizes = dataset_charts()
            with gr.Row():
                gr.Plot(ds_bars, show_label=False)
                gr.Plot(ds_sizes, show_label=False)
            with gr.Accordion("Dataset table", open=False):
                gr.Dataframe(dataset_table(), interactive=False, show_label=False)

    controls = [t_low, t_high, split, base_rate, cost_fn, cost_fp, seconds, eggs_day]
    outputs = [kpis, route, cm_raw, cm_1000, alts_fig, alts, curve]
    for c in controls:
        c.change(update, controls, outputs)
    for c in [img_in, t_low, t_high]:
        c.change(check_egg, [img_in, t_low, t_high], [img_out, verdict])
    reset.click(lambda: (T_LOW0, T_HIGH0), None, [t_low, t_high])
    demo.load(update, controls, outputs)
    demo.load(check_egg, [img_in, t_low, t_high], [img_out, verdict])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=THEME, css=CSS, head=HEAD)
