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
                         font={"family": "Segoe UI, sans-serif", "size": 13, "color": "#223943"},
                         colorway=["#006F83", "#395DA8", "#9A6A28", "#755CA4", "#477B68"],
                         paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=490,
                         margin={"l": 70, "r": 30, "t": 75, "b": 70})
    figure.update_xaxes(gridcolor="#E3EAED", linecolor="#B7C8CE", zerolinecolor="#B7C8CE")
    figure.update_yaxes(gridcolor="#E3EAED", linecolor="#B7C8CE", zerolinecolor="#B7C8CE")
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
    _base(evolution, "Evolution history (convergence uses confirmed full-CV frontier)")
    evolution.update_layout(height=920)
    evolution.update_xaxes(title_text="Generation", row=3, col=1)
    figures["history"] = evolution
    return figures


_REPORT_CSS = """
:root{color-scheme:light;--ink:#17313d;--muted:#516873;--paper:#eef3f4;--white:#fff;
  --line:#d4e0e4;--signal:#006f83;--deep:#102c39;--alert:#9b3e2c}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);
  color:var(--ink);font:16px/1.55 'Segoe UI',Arial,sans-serif}
a{color:var(--signal)}a:focus-visible{outline:3px solid #e4ad4b;outline-offset:3px}
.skip-link{position:absolute;top:-4rem;left:1rem;z-index:10;background:#fff;padding:.5rem 1rem}
.skip-link:focus{top:1rem}.masthead{background:var(--deep);color:#f4f8f9;
  border-bottom:6px solid var(--signal)}.wrap{max-width:1296px;margin:auto;padding-left:clamp(20px,4vw,56px);
  padding-right:clamp(20px,4vw,56px)}.masthead .wrap{padding-top:42px;padding-bottom:42px}
.identity{font:600 .85rem/1.2 Bahnschrift,'Arial Narrow',Arial,sans-serif;color:#a9c9d1;
  letter-spacing:.02em;margin:0 0 24px}.masthead h1{font:600 clamp(2.3rem,4.4vw,4.3rem)/1.03
  Bahnschrift,'Arial Narrow',Arial,sans-serif;letter-spacing:-.035em;max-width:24ch;margin:0 0 18px}
.masthead p{max-width:70ch;color:#c6d8de;margin:0}.overview{background:var(--white);
  border-bottom:1px solid var(--line)}.overview .wrap{display:grid;grid-template-columns:minmax(0,1.3fr)
  minmax(300px,.8fr);gap:clamp(24px,4vw,60px);padding-top:34px;padding-bottom:34px}
.overview h2{font:600 1.45rem/1.2 Bahnschrift,'Arial Narrow',Arial,sans-serif;margin:0 0 12px}
.overview p{max-width:70ch;margin:0 0 12px}.overview .caution{font-size:.91rem;color:var(--muted)}
.comparison{border-collapse:collapse;width:100%;font-size:.89rem;align-self:start}
.comparison caption{font-weight:650;text-align:left;padding:0 0 10px;color:var(--ink)}
.comparison th,.comparison td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}
.comparison th:first-child,.comparison td:first-child{text-align:left;padding-left:0}
.comparison thead th{font-weight:600;color:var(--muted);white-space:nowrap}
.comparison tbody th{font-weight:500;color:var(--muted)}.comparison td{font-variant-numeric:tabular-nums}
.contents{background:#e3ebee;border-bottom:1px solid #c8d7dc}.contents .wrap{display:flex;
  flex-wrap:wrap;gap:2px 22px;align-items:center;padding-top:11px;padding-bottom:11px}
.contents strong{font:600 .82rem Bahnschrift,'Arial Narrow',Arial,sans-serif;color:#526873;
  margin-right:3px}.contents a{text-decoration:none;font-size:.86rem;font-weight:600;padding:5px 0}
.contents a:hover{text-decoration:underline}main.wrap{padding-top:28px;padding-bottom:70px}
.report-section{scroll-margin-top:20px;background:var(--white);border:1px solid var(--line);
  margin:0 0 20px;padding:clamp(18px,2.3vw,32px)}.report-section.primary{border-top:5px solid var(--signal)}
.section-heading{display:flex;align-items:baseline;justify-content:space-between;gap:14px;
  border-bottom:1px solid var(--line);padding-bottom:13px;margin-bottom:12px}
.section-heading h2{font:600 clamp(1.35rem,2vw,1.8rem)/1.2 Bahnschrift,'Arial Narrow',Arial,sans-serif;
  letter-spacing:-.015em;margin:0}.section-heading a{font-size:.8rem;white-space:nowrap}
.section-intro{color:var(--muted);max-width:75ch;font-size:.92rem;margin:0 0 8px}
.chart{min-width:0}.chart .js-plotly-plot{width:100%}.feature-list{list-style:none;margin:18px 0 0;
  padding:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:0 18px}
.feature-list li{border-bottom:1px solid var(--line);padding:8px 0;overflow-wrap:anywhere}
.feature-list code,code{font: .9em/1.5 Consolas,'Courier New',monospace}
.table-scroll{overflow:auto;max-height:540px;border:1px solid var(--line);margin-top:16px}
.constraints{border-collapse:collapse;width:100%;font-size:.85rem;min-width:720px}
.constraints th,.constraints td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left}
.constraints thead th{position:sticky;top:0;background:#e8eff1;color:var(--ink);z-index:1}
.constraints tbody tr:nth-child(even){background:#f7f9fa}.constraints tr.violation{background:#fcece9;color:var(--alert)}
.constraints td:nth-child(n+4){font-variant-numeric:tabular-nums}
.footer{color:var(--muted);font-size:.82rem;border-top:1px solid var(--line);padding-top:20px}
@media(max-width:760px){.masthead .wrap{padding-top:29px;padding-bottom:29px}
  .overview .wrap{grid-template-columns:1fr;gap:18px}.report-section{padding:16px}
  .section-heading{align-items:flex-start}.feature-list{grid-template-columns:1fr 1fr}}
@media(max-width:520px){.feature-list{grid-template-columns:1fr}.contents .wrap{gap:0 14px}
  .comparison{font-size:.78rem}.comparison th,.comparison td{padding:7px 4px}
  .section-heading h2{font-size:1.3rem}.section-heading a{display:none}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{body{background:#fff}.masthead{background:#fff;color:#000;border-color:#000}
  .masthead p,.identity{color:#333}.contents,.section-heading a{display:none}
  .report-section{break-inside:avoid;border-color:#bbb}.table-scroll{max-height:none;overflow:visible}}
"""


def write_report(path: str | Path, payload: dict) -> Path:
    """Write one fully self-contained Plotly HTML artifact, usable without internet."""
    path = Path(path)
    figures = build_figures(payload)
    descriptions = {
        "frontier": ("Observed Pareto frontier", "Each point is an evaluated subset. The teal line joins confirmed feasible solutions; the star is the full-feature reference."),
        "cardinality": ("Performance by subset size", "Best observed full-validation Gini at each evaluated feature count. This is search evidence, not a complete sweep."),
        "frequency": ("Features in strong subsets", f"Inclusion frequency across distinct confirmed Pareto subsets within {payload.get('gini_tolerance', 0.005):g} Gini of the best observed result."),
        "coselection": ("Feature co-selection", "How often two features appear together in the confirmed Pareto set."),
        "temporal": ("Validation over time", "Fold-level Gini for the selected model, best confirmed result, reference, and other frontier subsets."),
        "segments": ("Segment performance", "Equal-fold mean Gini by segment. Any configured hard floors are checked per fold in the table below."),
        "history": ("Search history", "Confirmed frontier progress, evaluation counts, and population size across generations."),
    }
    sections = []
    for name, figure in figures.items():
        heading, description = descriptions[name]
        figure.update_layout(title=None, margin={"t": 35})
        chart = figure.to_html(full_html=False, include_plotlyjs=False, div_id="chart-" + name,
                               config={"responsive": True, "displaylogo": False})
        sections.append(f'<section class="report-section{" primary" if name == "frontier" else ""}" id="{name}">'
                        f'<div class="section-heading"><h2>{heading}</h2><a href="#top">Back to top</a></div>'
                        f'<p class="section-intro">{description}</p><div class="chart">{chart}</div></section>')
        if name == "frontier":
            feature_items = ''.join(f'<li><code>{escape(str(feature))}</code></li>' for feature in payload['best_features'])
            feature_body = (f'<ul class="feature-list">{feature_items}</ul>' if feature_items else
                            '<p>No features were selected in this saved result.</p>')
            sections.append('<section class="report-section" id="selected-features">'
                            '<div class="section-heading"><h2>Selected features</h2><a href="#top">Back to top</a></div>'
                            '<p class="section-intro">The feature set designated by this run for external validation.</p>'
                            f'{feature_body}</section>')
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
        constraint_body = (f'<p class="section-intro">{len(violations)} violated full-CV comparisons. Selected, best and reference comparisons are shown, '
                           'followed by up to 500 other violations. Complete comparisons, support counts and fidelity tags '
                           f'are in <code>segment_constraints.{escape(str(payload.get("output_format", "parquet")))}</code>.</p>'
                           '<div class="table-scroll"><table class="constraints"><thead><tr>' +
                           ''.join(f'<th scope="col">{h}</th>' for h in headings) +
                           '</tr></thead><tbody>' + ''.join(table_rows) + '</tbody></table></div>')
    else:
        constraint_body = ('<p class="section-intro">No active segment hard constraints were configured. '
                           'Any exclusions for insufficient event support are recorded in the experiment manifest.</p>')
    sections.append('<section class="report-section" id="constraint-table">'
                    '<div class="section-heading"><h2>Per-fold segment constraints</h2><a href="#top">Back to top</a></div>'
                    f'{constraint_body}</section>')

    names = payload["feature_names"]
    selected_mask = [name in payload["best_features"] for name in names]
    candidates = [r for r in payload["results"] if r["fidelity"] == "full"]
    selected_id = payload.get("selected_candidate_id")
    selected = next((r for r in candidates if r["key"] == selected_id), None) if selected_id else None
    if selected is None:
        selected = next((r for r in candidates if r["mask"] == selected_mask), None)
    reference = payload["reference"]
    def gini(result: dict | None) -> str:
        value = result.get("mean_gini") if result else None
        return f"{value:.4f}" if isinstance(value, (int, float)) else "—"

    converged = any(row.get("converged", False) for row in payload.get("history", []))
    links = [("frontier", "Frontier"), ("selected-features", "Selected features"),
             ("cardinality", "Subset size"), ("frequency", "Frequency"),
             ("coselection", "Co-selection"), ("temporal", "Over time"),
             ("segments", "Segments"), ("history", "Search history"),
             ("constraint-table", "Constraints")]
    navigation = ''.join(f'<a href="#{target}">{label}</a>' for target, label in links)
    text = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Feature selection | Development validation report</title><style>{_REPORT_CSS}</style>
<script>{get_plotlyjs()}</script></head><body id="top"><a class="skip-link" href="#report">Skip to report</a>
<header class="masthead"><div class="wrap"><p class="identity">Memetic NSGA-II / experiment report</p>
<h1>Feature selection under temporal validation</h1>
<p>Observed trade-offs from the development search. {len(payload['pareto'])} confirmed Pareto subsets; numerical convergence {'observed' if converged else 'not observed'}.</p>
</div></header><div class="overview"><div class="wrap"><div><h2>Read this result</h2>
<p>The selected subset is a candidate for an independent external OOT assessment. These plots describe evaluated subsets and do not establish a global optimum.</p>
<p class="caution">Repeated search creates selection optimism. Provisional fold-reduced measurements have a separate evaluation basis. Feature frequency and co-selection describe searched solution families, not statistical feature significance.</p></div>
<table class="comparison"><caption>Selected versus full-feature reference</caption><thead><tr><th scope="col">Measure</th><th scope="col">Selected</th><th scope="col">Reference</th></tr></thead>
<tbody><tr><th scope="row">Features</th><td>{len(payload['best_features'])} / {len(names)}</td><td>{reference['n_features']} / {len(names)}</td></tr>
<tr><th scope="row">Mean validation Gini</th><td>{gini(selected)}</td><td>{gini(reference)}</td></tr>
<tr><th scope="row">Evaluation</th><td>{'Full CV' if selected else 'Unavailable'}</td><td>Full CV</td></tr></tbody></table></div></div>
<nav class="contents" aria-label="Report sections"><div class="wrap"><strong>Explore</strong>{navigation}</div></nav>
<main class="wrap" id="report">{''.join(sections)}<p class="footer">Development validation evidence only. Review the saved tables and external OOT results before using a subset in production.</p></main>
<script>
(() => {{
  const charts = ['frontier', 'cardinality', 'temporal', 'segments', 'history'];
  const compact = matchMedia('(max-width: 640px)');
  function arrangeCharts() {{
    for (const name of charts) {{
      const plot = document.getElementById('chart-' + name);
      if (!plot || !plot.data) continue;
      const small = compact.matches;
      Plotly.relayout(plot, {{
        legend: small ? {{orientation:'h', x:0, y:-0.26, font:{{size:11}}}} :
                        {{orientation:'v', x:1.02, y:1, font:{{size:12}}}},
        margin: small ? {{l:55, r:12, t:25, b:name === 'history' ? 210 : 130}} :
                        {{l:70, r:30, t:35, b:70}}
      }});
    }}
    const frequency = document.getElementById('chart-frequency');
    if (frequency && frequency.data) Plotly.relayout(frequency, {{showlegend:!compact.matches}});
  }}
  arrangeCharts();
  compact.addEventListener('change', arrangeCharts);
}})();
</script></body></html>"""
    path.write_text(text, encoding="utf-8")
    return path.resolve()
