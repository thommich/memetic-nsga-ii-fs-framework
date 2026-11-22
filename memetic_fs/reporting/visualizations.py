"""Plotly diagnostics built exclusively from standardized compact metrics."""
from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots


def segment_constraint_comparisons(payload: dict) -> list[dict]:
    """Compute explicit per-fold segment floors and violations for saved results.

    Reference and candidate results share canonical folds; provisional results
    compare only their selected folds. Undefined descriptive metrics remain
    undefined and are never labelled feasible by this diagnostic helper.
    """
    reference = payload["reference"]
    rows = []
    for result in payload.get("results", []):
        for constraint in payload.get("segment_constraints", []):
            for metric in result.get("segments", []):
                if metric["column"] != constraint["column"] or metric["value"] != constraint["value"]:
                    continue
                baseline = next((r for r in reference.get("segments", []) if r["column"] == metric["column"] and
                                 r["value"] == metric["value"] and r["fold"] == metric["fold"]), None)
                reference_gini = baseline["gini"] if baseline else None
                floor = reference_gini - constraint["max_gini_drop"] if reference_gini is not None else None
                violation = floor - metric["gini"] if floor is not None and metric["gini"] is not None else None
                rows.append({"key": result["key"], "fidelity": result["fidelity"], "column": metric["column"],
                             "value": metric["value"], "fold": metric["fold"], "candidate_gini": metric["gini"],
                             "reference_gini": reference_gini, "minimum_gini": floor,
                             "max_gini_drop": constraint["max_gini_drop"], "violation": violation,
                             "violated": violation > 0 if violation is not None else None,
                             "n": metric["n"], "events": metric["events"], "nonevents": metric["nonevents"]})
    return rows


def _base(figure: go.Figure, title: str, x: str = "", y: str = "") -> go.Figure:
    figure.update_layout(title=title, xaxis_title=x, yaxis_title=y, template="plotly_white",
                         font={"family": "Arial, sans-serif", "size": 13}, height=490,
                         margin={"l": 70, "r": 30, "t": 85, "b": 70})
    return figure


def build_figures(payload: dict, max_cofeatures: int = 30) -> dict[str, go.Figure]:
    """Create aligned full/provisional charts and a top-feature co-selection map."""
    from .exports import selection_frequency

    results, front = payload["results"], payload["pareto"]
    reference, names = payload["reference"], payload["feature_names"]
    confirmed = list({tuple(r["mask"]): r for r in results if r["fidelity"] == "full" and r["key"] != reference["key"]}.values())
    provisional = [r for r in results if r["fidelity"] != "full"]
    frontier = go.Figure()
    for label, values, marker in [("Confirmed evaluated subsets", confirmed, {"color": "#9CA3AF", "size": 7}),
                                  ("Provisional (reduced folds)", provisional, {"color": "#E3A03A", "symbol": "x", "size": 7}),
                                  ("Confirmed feasible Pareto subsets", front, {"color": "#007C91", "size": 11})]:
        frontier.add_trace(go.Scatter(x=[r["n_features"] for r in values], y=[r["mean_gini"] for r in values],
            mode="markers+lines" if "Pareto" in label else "markers", name=label, marker=marker,
            customdata=[[r["key"], r.get("std_gini"), r["fidelity"], max([0.] + r.get("constraints", [])),
                         r.get("min_gini"), len(r.get("folds", [])), sum(f.get("n", 0) for f in r.get("folds", []))] for r in values],
            hovertemplate="%{customdata[0]}<br>Features: %{x}<br>Mean Gini: %{y:.5f}<br>Fold SD: %{customdata[1]:.5f}<br>Minimum fold Gini: %{customdata[4]:.5f}<br>Validation folds: %{customdata[5]}<br>Validation observations: %{customdata[6]}<br>Fidelity: %{customdata[2]}<br>Constraint violation: %{customdata[3]:.5f}<extra>%{fullData.name}</extra>"))
    frontier.add_trace(go.Scatter(x=[reference["n_features"]], y=[reference["mean_gini"]], name="Full-feature reference",
                                  mode="markers", marker={"symbol": "star", "size": 16, "color": "#7B3294"}))
    figures = {"frontier": _base(frontier, "Observed development Pareto frontier", "Selected features", "Mean validation-fold Gini")}

    best_by_count = {}
    for result in confirmed:
        k = result["n_features"]
        best_by_count[k] = max(result["mean_gini"], best_by_count.get(k, -1.))
    counts = sorted(best_by_count)
    count_chart = go.Figure(go.Scatter(x=counts, y=[best_by_count[k] for k in counts], mode="lines+markers", name="Best observed full-CV Gini"))
    rfe = sorted(payload.get("rfe_trajectory", []), key=lambda r: r["n_features"])
    if rfe:
        count_chart.add_trace(go.Scatter(x=[r["n_features"] for r in rfe], y=[r["mean_gini"] for r in rfe], mode="lines+markers", name="RFE initialization"))
    figures["cardinality"] = _base(count_chart, "Best discovered Gini by evaluated cardinality (search evidence)", "Selected features", "Mean validation-fold Gini")

    tolerance = payload.get("gini_tolerance", 0.005)
    high = [r for r in front if r["mean_gini"] >= max((v["mean_gini"] for v in front), default=0) - tolerance]
    frequency = selection_frequency(high, names, payload.get("feature_groups")).head(40).iloc[::-1]
    figures["frequency"] = _base(go.Figure(go.Bar(x=frequency.frequency, y=frequency.feature, orientation="h", name="Inclusion frequency")),
                                f"Feature frequency within {tolerance:g} Gini of the best Pareto subset", "Fraction of distinct subsets", "")
    figures["frequency"].update_layout(height=max(490, len(frequency) * 21))
    ranked = selection_frequency(front, names).head(max_cofeatures).feature.tolist()
    indices = [names.index(name) for name in ranked]
    masks = np.asarray([r["mask"] for r in front], dtype=float)
    co = masks[:, indices].T @ masks[:, indices] / len(front) if len(front) else np.zeros((len(indices), len(indices)))
    figures["coselection"] = _base(go.Figure(go.Heatmap(x=ranked, y=ranked, z=co, zmin=0, zmax=1,
            colorscale="Blues", colorbar={"title": "Fraction"}, hovertemplate="%{x} + %{y}<br>Co-selection: %{z:.3f}<extra></extra>")),
            f"Co-selection across confirmed Pareto subsets (top {max_cofeatures})")
    figures["coselection"].update_layout(height=740, xaxis_tickangle=-50, margin={"b": 160, "l": 160})

    selected_mask = [name in payload["best_features"] for name in names]
    selected_id = payload.get("selected_candidate_id")
    selected = next((r for r in confirmed + front if r["key"] == selected_id), None) if selected_id else next((r for r in confirmed + front if r["mask"] == selected_mask), None)
    best = max(front, key=lambda r: r["mean_gini"], default=None)
    designated = [("Selected model", selected), ("Best confirmed Gini", best), ("Full-feature reference", reference)]
    for result in front:
        if len(designated) >= 7:
            break
        if result["key"] not in {r["key"] for _, r in designated if r}:
            designated.append((f"Pareto: {result['n_features']} features", result))
    temporal = go.Figure()
    for label, result in designated:
        if result:
            temporal.add_trace(go.Scatter(x=[f["fold"] for f in result["folds"]], y=[f["gini"] for f in result["folds"]],
                mode="lines+markers", name=label,
                hovertemplate="%{x}<br>Validation Gini: %{y:.5f}<extra>%{fullData.name}</extra>"))
    figures["temporal"] = _base(temporal, "Temporal development-validation stability", "Validation interval", "Gini")

    segment = go.Figure()
    constraints = payload.get("segment_constraints", [])
    for label, result in designated:
        if not result:
            continue
        groups = {}
        for value in result.get("segments", []):
            name = f"{value['column']}={value['value']}"
            if value["gini"] is not None:
                groups.setdefault(name, []).append(value["gini"])
        segment.add_trace(go.Bar(x=list(groups), y=[np.mean(values) for values in groups.values()], name=label))
    if constraints:
        reference_groups = {}
        for value in reference.get("segments", []):
            if value["gini"] is not None:
                reference_groups.setdefault(f"{value['column']}={value['value']}", []).append(value["gini"])
        for constraint in constraints:
            name = f"{constraint['column']}={constraint['value']}"
            if name in reference_groups:
                threshold = np.mean(reference_groups[name]) - constraint["max_gini_drop"]
                segment.add_trace(go.Scatter(x=[name], y=[threshold], mode="markers", name=f"Mean constraint floor: {name}",
                                            marker={"symbol": "line-ew", "size": 25, "color": "#A63232"}))
    segment.update_layout(barmode="group")
    if not any(r and r.get("segments") for _, r in designated):
        segment.add_annotation(text="No business-segment metadata was configured.", x=0.5, y=0.5,
                               xref="paper", yref="paper", showarrow=False)
    figures["segments"] = _base(segment, "Segment validation Gini (equal fold means; constraints apply per fold)", "Segment", "Mean defined fold Gini")

    history = payload.get("history", [])
    evolution = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                              subplot_titles=["Confirmed frontier progress", "Evaluation and archive counts", "Population cardinality"])
    generations = [h["generation"] for h in history]
    for fidelity in sorted({h["fidelity"] for h in history}):
        rows = [h for h in history if h["fidelity"] == fidelity]
        evolution.add_trace(go.Scatter(x=[h["generation"] for h in rows], y=[h.get("best_confirmed_gini") for h in rows],
            name=f"Confirmed Gini; exploration={fidelity}", mode="lines+markers"), row=1, col=1)
    evolution.add_trace(go.Scatter(x=generations, y=[h.get("confirmed_hypervolume") for h in history], name="Confirmed normalized hypervolume"), row=1, col=1)
    for key, label in [("unique_evaluations", "Unique evaluation configurations"), ("confirmed_count", "Confirmed Pareto count"), ("provisional_count", "Provisional Pareto count")]:
        evolution.add_trace(go.Scatter(x=generations, y=[h.get(key) for h in history], name=label), row=2, col=1)
    for key in ["feature_count_min", "feature_count_mean", "feature_count_max"]:
        evolution.add_trace(go.Scatter(x=generations, y=[h.get(key) for h in history], name=key.replace("_", " ")), row=3, col=1)
    evolution.update_layout(height=920, title="Evolution history (convergence uses confirmed full-CV frontier)", template="plotly_white")
    evolution.update_xaxes(title_text="Generation", row=3, col=1)
    figures["history"] = evolution
    return figures


def write_report(path: str | Path, payload: dict) -> Path:
    """Write one fully self-contained Plotly HTML artifact, usable without internet."""
    path = Path(path)
    figures = build_figures(payload)
    sections = []
    for name, figure in figures.items():
        sections.append(f'<section id="{escape(name)}">{figure.to_html(full_html=False, include_plotlyjs=False, div_id="chart-" + name)}</section>')
    constraints = segment_constraint_comparisons(payload)
    if constraints:
        selected_mask = [name in payload["best_features"] for name in payload["feature_names"]]
        designated = {r["key"] for r in payload.get("pareto", []) if r["mask"] == selected_mask}
        best = max(payload.get("pareto", []), key=lambda r: r["mean_gini"], default=None)
        if best:
            designated.add(best["key"])
        designated.add(payload["reference"]["key"])
        display = [row for row in constraints if row["fidelity"] == "full" and row["key"] in designated]
        violations = [row for row in constraints if row["fidelity"] == "full" and row["violated"]]
        display += [row for row in violations if row["key"] not in designated][:500]
        headings = ["Candidate", "Segment", "Fold", "Gini", "Required minimum", "Violation", "Status"]
        table_rows = []
        for row in display:
            values = [row["key"][:14], f"{row['column']}={row['value']}", row["fold"],
                      row["candidate_gini"], row["minimum_gini"], row["violation"],
                      "VIOLATION" if row["violated"] else ("UNDEFINED" if row["violated"] is None else "Pass")]
            cells = "".join(f"<td>{escape(f'{value:.5f}' if isinstance(value, float) else str(value))}</td>" for value in values)
            table_rows.append(f'<tr class="{"violation" if row["violated"] else ""}">{cells}</tr>')
        sections.append('<section id="constraint-table"><h2>Per-fold segment constraints</h2>'
                        f'<p>{len(violations)} violated full-CV comparisons. Selected, best and reference comparisons are shown, '
                        'followed by up to 500 other violations. Complete comparisons, support counts and explicit fidelity tags '
                        f'are available in <code>segment_constraints.{payload.get("output_format", "parquet")}</code>.</p>'
                        '<div style="overflow:auto"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headings) +
                        '</tr></thead><tbody>' + ''.join(table_rows) + '</tbody></table></div></section>')
    else:
        sections.append('<section id="constraint-table"><h2>Per-fold segment constraints</h2><p>No active segment hard constraints were configured. '
                        'Any exclusions for insufficient event support are recorded in the experiment manifest.</p></section>')
    converged = any(row.get("converged", False) for row in payload.get("history", []))
    text = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memetic feature selection experiment</title><style>body{{margin:0;background:#f4f6f8;color:#182934;font:16px Arial,sans-serif}}main{{max-width:1250px;margin:auto;padding:32px}}section{{background:white;margin:24px 0;border-radius:8px;padding:12px}}p{{line-height:1.6}}nav a{{margin-right:16px;color:#007c91}}code{{font-size:13px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #d8e0e5}}tr.violation{{background:#fff0ef;color:#9b2525}}th{{background:#edf3f6}}</style>
<script>{get_plotlyjs()}</script></head><body><main><h1>Feature subset selection</h1>
<p>Development temporal validation. {len(payload['pareto'])} confirmed Pareto subsets; {len(payload['best_features'])} selected features.
Numerical convergence observed: {str(converged).lower()}.</p>
<p>These charts describe evaluated subsets. They do not establish a global optimum. Repeated search creates selection optimism;
an independent external OOT assessment remains necessary. Provisional fold-reduced measurements have a separate evaluation basis.
Frequency and co-selection describe searched solution families, not statistical feature significance.</p>
<nav>{''.join(f'<a href="#{name}">{name.capitalize()}</a>' for name in figures)}</nav>
{''.join(sections)}<section><h2>Selected features</h2><p>{escape(', '.join(payload['best_features']))}</p></section></main></body></html>"""
    path.write_text(text, encoding="utf-8")
    return path.resolve()
