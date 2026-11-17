"""SignalForge AI - Product Intelligence & Decision OS (Streamlit UI). Domain logic lives in src/signalforge."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from signalforge.audit import AuditStore, Overrides
from signalforge.copilot import CopilotState, answer, llm_available, narrate_with_llm, sanitize_evidence
from signalforge.datasets import REGISTRY, coerce_to_schema, data_quality, load_public
from signalforge.economics import Assumptions, monte_carlo, tornado
from signalforge.exports import (
    generate_decision_log,
    generate_evidence_appendix,
    generate_experiment_plan,
    generate_memo,
    generate_opportunity_assessment,
    generate_prd,
    generate_roadmap_md,
    markdown_to_docx,
)
from signalforge.ingestion import MAX_BYTES, REQUIRED, IngestionError, safe_csv, validate
from signalforge.intelligence import cluster_feedback, near_duplicates, suggest_merges
from signalforge.optimizer import Constraints, alternative, greedy_baseline, optimise
from signalforge.pipeline import build_opportunities, explain_outranks, portfolio_total, rank_opportunities, scenario_ranks
from signalforge.prioritization import FRAMEWORKS, Weights, compare_frameworks, score, weight_sensitivity
from signalforge.synthetic import generate

st.set_page_config(page_title="SignalForge AI", page_icon="◆", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.2rem;max-width:1360px}
h1,h2,h3{letter-spacing:-.01em}
[data-testid=stMetricValue]{font-size:1.45rem;font-weight:650}
[data-testid=stMetricLabel]{color:#8a94a6}
.sf-pill{display:inline-block;padding:.12rem .55rem;border-radius:999px;font-size:.74rem;font-weight:600;margin-right:.35rem}
.sf-syn{background:#fff4d6;color:#7a5600}.sf-real{background:#ddf4e4;color:#165c2e}.sf-warn{background:#fde2e1;color:#8a1c1c}
.sf-banner{border-left:4px solid #6c5ce7;background:rgba(108,92,231,.07);padding:.7rem 1rem;border-radius:8px;margin:.4rem 0 1rem}
.sf-banner b{color:#6c5ce7}
.sf-muted{color:#8a94a6;font-size:.85rem}
</style>""", unsafe_allow_html=True)
fmt = lambda v: "n/a" if v is None or pd.isna(v) else f"${v:,.0f}"  # noqa: E731

# ---------------- per-visitor state (in-memory; never shared, never written to disk) ----------------
ss = st.session_state
if "store" not in ss:
    ss.store = AuditStore(":memory:")
    ss.chat = []
    ss.onboarded = False
store: AuditStore = ss.store


@st.cache_data(show_spinner="Clustering feedback (semantic, deterministic)...", max_entries=8)
def run_clustering(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    return cluster_feedback(df, thr)


@st.cache_data(show_spinner="Fetching public dataset from its original host...", ttl=3600, max_entries=6)
def fetch_public(key: str, repo: str | None) -> pd.DataFrame:
    return load_public(key, repo)  # cached per (dataset, repo): public data only, never user uploads


# ---------------- sidebar: data ----------------
st.sidebar.markdown("### ◆ SignalForge AI")
st.sidebar.caption("Your customers are telling you what to build. SignalForge tells you what deserves the budget.")
SOURCES = {"Synthetic demo (1,000 records)": "synthetic", "Public: F-Droid app reviews": "fdroid_reviews",
           "Public: Bitext support requests": "bitext_support", "Public: GitHub issues (live)": "github_issues", "Upload file": "upload"}
src_label = st.sidebar.selectbox("Dataset", list(SOURCES), key="dataset")
src = SOURCES[src_label]
st.sidebar.download_button("Download CSV template", ",".join(REQUIRED) + "\n", "signalforge_template.csv", "text/csv")

ingest, notes = None, []
try:
    if src == "synthetic":
        ingest = validate(generate())
    elif src == "upload":
        up = st.sidebar.file_uploader("Feedback file (CSV, JSON, JSONL, XLSX; max 10 MB)", type=["csv", "json", "jsonl", "xlsx"])
        if up is None:
            st.markdown("## No data, no drama.\nUpload feedback and let's find the signal. Any table with a text column works; "
                        "SignalForge maps columns automatically and tells you exactly what it assumed. Don't upload personal "
                        "data you aren't authorised to process.")
            st.stop()
        raw = up.getvalue()
        if len(raw) > MAX_BYTES:
            raise IngestionError("That file is over 10 MB. Split it or sample it, then try again.")
        name = up.name.lower()
        import io

        if name.endswith(".csv"):
            df_in = pd.read_csv(io.BytesIO(raw), encoding_errors="replace", on_bad_lines="skip", engine="python")
        elif name.endswith(".jsonl"):
            df_in = pd.read_json(io.BytesIO(raw), lines=True)
        elif name.endswith(".json"):
            df_in = pd.read_json(io.BytesIO(raw))
        else:
            df_in = pd.read_excel(io.BytesIO(raw))
        if df_in.empty:
            raise IngestionError("That file has no rows. Here's exactly what needs fixing: add at least 3 feedback records.")
        df_in, notes = coerce_to_schema(df_in)
        ingest = validate(df_in)
    else:
        repo = None
        if src == "github_issues":
            repo = st.sidebar.text_input("Public repo (owner/name)", "streamlit/streamlit")
        ingest = validate(fetch_public(src, repo))
except IngestionError as e:
    st.error(f"That file isn't cooperating. {e}")
    st.stop()
except Exception as e:  # noqa: BLE001 - malformed uploads must never crash the app
    st.error(f"That file isn't cooperating ({type(e).__name__}). Check it's valid CSV/JSON/XLSX with a header row.")
    st.stop()

spec = REGISTRY.get(src)
is_synth = src == "synthetic"
has_acv = ingest.df["annual_contract_value"].notna().any()

st.sidebar.markdown("---")
thr = st.sidebar.slider("Cluster breadth", 0.5, 0.98, 0.9, 0.01, help="Cosine distance threshold. Higher = fewer, broader themes.", key="thr")
with st.sidebar.expander("Financial assumptions", expanded=not has_acv):
    if not has_acv:
        st.caption("This dataset has **no revenue**. Values below are labelled user assumptions, not facts.")
    a = Assumptions(
        base_exposure_fraction=st.slider("Exposure fraction of ACV", 0.0, 1.0, 0.30, 0.05, key="exp"),
        recovery_probability=st.slider("Recovery probability", 0.0, 1.0, 0.40, 0.05, key="rec"),
        eng_cost_per_person_week=st.number_input("Eng cost / person-week (USD)", 0.0, 50000.0, 4000.0, 500.0, key="eng"),
        infra_cost_annual=st.number_input("Incremental infra / yr (USD)", 0.0, 1e6, 2000.0, 500.0, key="infra"),
        support_cost_annual=st.number_input("Incremental support / yr (USD)", 0.0, 1e6, 3000.0, 500.0, key="sup"),
        impute_missing_acv=st.checkbox("Impute missing ACV (segment median)", True, key="impute"),
        assumed_acv_per_account=(st.number_input("ASSUMED value per affected account (USD/yr)", 0.0, 1e6, 500.0, 50.0, key="assumed")
                                 if not has_acv else None),
    ).validated()
if st.sidebar.button("Reset my session (data, overrides, audit)", key="reset"):
    store.reset(); ss.chat = []
    st.rerun()

# ---------------- analysis ----------------
clustered = run_clustering(ingest.df, thr)
sig = (src, len(ingest.df), thr, str(ingest.df["ticket_id"].iloc[0]))
if ss.get("sig") != sig:  # cluster ids change with data/threshold: stale overrides must not silently carry over
    if ss.get("sig") is not None:
        store.save_overrides(Overrides()); store.log("overrides_cleared", "Data source or threshold changed")
    ss["sig"] = sig
ov: Overrides = store.load_overrides()
opps = build_opportunities(clustered, a, ov)
ranked = rank_opportunities(opps, a, ov)
by_id = {o["cluster_id"]: o for o in opps}
dq = data_quality(ingest.df)
if ranked.empty:
    st.warning("This theme needs more evidence before it earns roadmap space: no actionable cluster has 3+ records. "
               "Try a broader cluster setting or more data.")
    st.stop()

pill = ('<span class="sf-pill sf-syn">SYNTHETIC DEMO - fictional accounts & revenue</span>' if is_synth else
        f'<span class="sf-pill sf-real">REAL PUBLIC DATA - {spec.name}</span>' if spec else '<span class="sf-pill sf-real">YOUR UPLOAD</span>')
if not has_acv:
    pill += '<span class="sf-pill sf-warn">NO REVENUE DATA - $ values use your assumption</span>'
st.markdown(pill, unsafe_allow_html=True)

PAGES = ["Executive Dashboard", "Customer Intelligence", "Opportunity Economics", "Roadmap Optimizer", "Scenario Lab", "Decision Memo & PRD"]
page = st.sidebar.radio("Navigate", PAGES, key="page")

# ---------------- roadmap constraints (shared by optimizer, copilot, memo) ----------------
cap_default = float(round(ranked["effort_weeks"].sum() * 0.6, 1)) or 10.0
cons = Constraints(capacity_weeks=ss.get("cap", cap_default), min_confidence=ss.get("minconf", 0.0),
                   mandatory=ss.get("mandatory", []), dependencies=ss.get("deps", []),
                   budget_usd=ss.get("budget") or None)
plan = optimise(ranked, cons)


# ---------------- SIGNAL copilot (global, page-aware, session-scoped) ----------------
def signal_panel(selected: int | None = None) -> None:
    with st.expander("◆ SIGNAL - your AI product strategist", expanded=bool(ss.chat)):
        st.caption("Grounded in this session's data and calculations. " +
                   ("LLM narration on (server-side key)." if llm_available() else "Deterministic analytics (no LLM key needed)."))
        quick = ["What should we build next?", "Which three issues threaten the most revenue?", "Why is this cluster ranked first?",
                 "What happens if engineering capacity drops 30%?", "What would a CFO prioritize?", "Which recommendations have weak evidence?",
                 "What are we ignoring?", "Challenge this roadmap.", "What assumptions would reverse this decision?", "Run the decision debate"]
        cols = st.columns(5)
        clicked = None
        for i, qq in enumerate(quick):
            if cols[i % 5].button(qq, key=f"q_{page}_{i}", width="stretch"):
                clicked = qq
        q = st.text_input("Ask SIGNAL", key=f"ask_{page}", placeholder="e.g. Which feature should we kill?")
        if st.button("Ask", key=f"askbtn_{page}") and q:
            clicked = q
        if clicked:
            stt = CopilotState(ranked, by_id, a, cons, page, selected, "synthetic" if is_synth else "real")
            rec = answer(stt, clicked)
            text = narrate_with_llm(rec) if llm_available() else None
            ss.chat.insert(0, (clicked, text or rec.to_markdown(), rec.table))
            store.log("copilot", f"[{rec.tool}] {clicked[:120]}")
        for qq, md, table in ss.chat[:3]:
            st.markdown(f"> **You:** {qq}")
            st.markdown(md)
            if table:
                st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)


def insight_banner() -> str:
    r0 = ranked.iloc[0]
    hs = ranked[(ranked.avg_severity >= 4) & (ranked.final_rank > 3)]
    if not hs.empty:
        x = hs.iloc[0]
        return (f"<b>SIGNAL:</b> '{x.label}' hits {x.unique_accounts} accounts at severity {x.avg_severity:.1f} and it's ranked "
                f"#{int(x.final_rank)}? Let's inspect the scoring assumptions.")
    if r0.evidence_sufficiency != "sufficient":
        return f"<b>SIGNAL:</b> Your #1 rests on {r0.tickets} records. That's a hypothesis, not a roadmap."
    return (f"<b>SIGNAL:</b> '{r0.label}': {r0.unique_accounts} accounts, {fmt(r0.net_opportunity_value)} modeled net value, "
            f"{r0.effort_weeks:.1f} person-weeks. This one deserves executive attention.")


# ================= PAGE 1 =================
if page == "Executive Dashboard":
    st.title("Executive Dashboard")
    signal_panel()
    if not ss.onboarded:
        with st.container(border=True):
            st.markdown("**First time here?** 1) Pick a dataset in the sidebar (synthetic demo or real public data). "
                        "2) Review themes in *Customer Intelligence*. 3) Tune assumptions and see *Economics* recalculate. "
                        "4) Set capacity in *Roadmap Optimizer*. 5) Ask SIGNAL to challenge it. 6) Export the memo.")
            if st.button("Got it", key="onb"):
                ss.onboarded = True; st.rerun()
    st.markdown(f'<div class="sf-banner">{insight_banner()}</div>', unsafe_allow_html=True)
    c = st.columns(6)
    c[0].metric("Records analysed", f"{len(ingest.df):,}", help=f"{ingest.dropped_rows} rejected, {ingest.duplicates_removed} exact duplicates removed")
    c[1].metric("Unique accounts", f"{ingest.df.account_id.nunique():,}")
    c[2].metric("Actionable themes", len(ranked), help=f"{sum(not o['actionable'] for o in opps)} non-actionable chatter clusters excluded")
    c[3].metric("Exposure (scenario)", fmt(portfolio_total([o for o in opps if o["actionable"]], a)),
                help="Each account counted once at its max exposure fraction. Scenario estimate, not churn.")
    c[4].metric("Roadmap value (optimised)", fmt(plan.total_value), help=f"{len(plan.selected)} items in {cons.capacity_weeks:.0f} person-weeks")
    c[5].metric("Data quality", f"{dq['score']:.0f}/100", help=str({k: f'{v:.0%}' for k, v in dq['components'].items()}))
    for w in ingest.warnings + notes:
        st.caption(f"Data note: {w}")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Recommended executive actions")
        for r in ranked.head(3).itertuples():
            with st.container(border=True):
                badge = {"sufficient": "🟢", "directional": "🟡", "insufficient": "🔴"}[r.evidence_sufficiency]
                st.markdown(f"**#{r.final_rank} · {r.label}** · {r.product_area} · {badge} evidence {r.evidence_sufficiency}")
                st.write(f"{r.unique_accounts} accounts · exposure {fmt(r.potential_revenue_exposure)} · net {fmt(r.net_opportunity_value)} · "
                         f"{r.effort_weeks:.1f} pw · confidence {r.confidence:.2f} · trend {r.trend_status}")
        if len(ranked) > 1:
            st.info(explain_outranks(ranked.iloc[0], ranked.iloc[1]))
    with right:
        st.subheader("Top product risks")
        risk = ranked.sort_values("potential_revenue_exposure", ascending=False).head(6)
        st.plotly_chart(px.bar(risk, x="potential_revenue_exposure", y="label", orientation="h", color="avg_severity",
                               color_continuous_scale="OrRd", labels={"potential_revenue_exposure": "Exposure (USD)", "label": ""})
                        .update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis={"autorange": "reversed"}),
                        width="stretch")
        em = ranked[ranked.trend_status.isin(["emerging", "rising"])]
        st.subheader("Emerging issues")
        st.write(", ".join(f"{x.label} ({x.trend_status})" for x in em.itertuples()) or "None detected (or not enough dated records).")
    st.caption("Every KPI's calculation is in Opportunity Economics; every theme's evidence is in Customer Intelligence.")

# ================= PAGE 2 =================
elif page == "Customer Intelligence":
    st.title("Customer Intelligence")
    signal_panel(ss.get("theme"))
    st.caption(f"Embedding backend: {clustered.attrs.get('backend', 'tfidf')} · {clustered.cluster_id.nunique()} raw clusters · "
               f"{int(clustered.is_outlier.sum())} outlier records")
    f1, f2, f3 = st.columns(3)
    q = f1.text_input("Search feedback", key="search")
    segs = f2.multiselect("Segment", sorted(clustered.customer_segment.unique()), key="segf")
    sevs = f3.slider("Min severity", 1, 5, 1, key="sevf")
    view = clustered[clustered.severity >= sevs]
    if q:
        view = view[view.feedback_text.str.contains(q, case=False, regex=False)]
    if segs:
        view = view[view.customer_segment.isin(segs)]
    st.dataframe(view[["ticket_id", "account_id", "customer_segment", "severity", "cluster_label", "feedback_text"]],
                 width="stretch", height=240, hide_index=True)
    st.download_button("Download processed CSV (formula-safe)", safe_csv(clustered), "signalforge_processed.csv", "text/csv")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(clustered, x="severity", color="customer_segment", title="Severity by segment"), width="stretch")
    sig_rows = [(o["label"], k, v) for o in opps if o["actionable"] for k, v in o["signal_profile"].items()]
    if sig_rows:
        c2.plotly_chart(px.density_heatmap(pd.DataFrame(sig_rows, columns=["theme", "signal", "share"]), x="signal", y="theme", z="share",
                                           histfunc="sum", title="Signal profile by theme", color_continuous_scale="Purples"),
                        width="stretch")

    st.subheader("Theme explorer: why these records were grouped")
    cid = st.selectbox("Theme", ranked.cluster_id, format_func=lambda i: f"#{int(ranked.set_index('cluster_id').final_rank[i])} {by_id[i]['label']}", key="theme")
    o = by_id[cid]
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Records", o["tickets"]); e2.metric("Unique accounts", o["unique_accounts"])
    e3.metric("Evidence", o["evidence_sufficiency"]); e4.metric("Trend", o["trend"]["status"])
    st.markdown(f"**Summary:** {o['unique_accounts']} accounts report problems in *{o['product_area']}* (avg severity "
                f"{o['avg_severity']:.1f}/5). Dominant signals: {', '.join(list(o['signal_profile'])[:3]) or 'none'}.")
    st.markdown("**Shared terms (share of records):** " + ", ".join(f"`{w}` {s:.0%}" for w, s in o["shared_terms"]))
    st.markdown("**Representative records** (closest to the theme centroid):")
    for r in o["representatives"]:
        txt, flagged = sanitize_evidence(r["text"])
        st.markdown(f"- `{r['ticket_id']}` (sim {r['similarity']:.2f}) {txt[:240]}" + (" ⚠️ *instruction-like text redacted*" if flagged else ""))
    if o["contradictions"]:
        st.markdown(f"**Contradicting evidence:** {', '.join(o['contradictions'])}")
    st.markdown(f"**Model confidence** {o['confidence']:.2f} (cluster tightness + breadth) · **Evidence sufficiency** "
                f"{o['evidence_sufficiency']} · **Business impact uncertainty**: depends on {'observed' if o['accounts_observed'] else 'assumed/unknown'} ACV.")
    with st.expander("All evidence records & affected accounts"):
        ev = clustered[clustered.ticket_id.isin(o["ticket_ids"])][["ticket_id", "account_id", "customer_segment", "severity", "feedback_text"]]
        st.dataframe(ev, width="stretch", height=240, hide_index=True)
        st.dataframe(pd.DataFrame(list(o["account_acv"].items()), columns=["account_id", "ACV (resolved)"]), width="stretch", hide_index=True)

    st.subheader("Human review")
    st.caption("Overrides are reversible, audited, and survive assumption changes (they reset only if the data or cluster setting changes).")
    h1, h2, h3 = st.columns(3)
    new = h1.text_input("Rename theme", key="rename")
    if h1.button("Apply rename", key="b_rename") and new:
        ov.rename[cid] = new[:80]; store.save_overrides(ov); store.log("rename", f"cluster {cid} -> {new[:80]}"); st.rerun()
    others = [i for i in ranked.cluster_id if i != cid]
    if others:
        other = h2.selectbox("Merge into this theme", others, format_func=lambda i: by_id[i]["label"], key="mergesel")
        if h2.button("Merge", key="b_merge"):
            ov.merge.append([cid, other]); store.save_overrides(ov); store.log("merge", f"{other} merged into {cid}"); st.rerun()
    if h3.button("Reject theme (not a real problem)", key="b_reject"):
        ov.rejected.append(int(cid)); store.save_overrides(ov); store.log("reject", f"cluster {cid} rejected"); st.rerun()
    sp = st.multiselect("Select records to split out (or exclude)", o["ticket_ids"], key="splitsel")
    s1, s2, s3 = st.columns(3)
    if s1.button("Split into new theme", key="b_split") and sp:
        new_id = int(max(clustered.cluster_id.max(), max(by_id)) + 1000 + len(ov.split))
        ov.split.append({"cluster_id": int(cid), "ticket_ids": sp, "new_id": new_id}); store.save_overrides(ov)
        store.log("split", f"{len(sp)} records split from {cid} -> {new_id}"); st.rerun()
    if s2.button("Exclude as irrelevant", key="b_excl") and sp:
        ov.irrelevant += sp; store.save_overrides(ov); store.log("irrelevant", f"{len(sp)} records excluded from {cid}"); st.rerun()
    if s3.button("Undo all overrides", key="b_undo"):
        store.save_overrides(Overrides()); store.log("overrides_cleared", "User undid all overrides"); st.rerun()
    st.subheader("Suggested merges & duplicates")
    sm = suggest_merges(clustered)
    lab = clustered.drop_duplicates("cluster_id").set_index("cluster_id")["cluster_label"]
    st.dataframe(sm.assign(label_a=sm.cluster_a.map(lab), label_b=sm.cluster_b.map(lab)) if not sm.empty else sm, width="stretch", hide_index=True)
    nd = near_duplicates(ingest.df) if len(ingest.df) <= 3000 else pd.DataFrame()
    st.write(f"Exact duplicates removed at ingestion: **{ingest.duplicates_removed}** · near-duplicate same-account pairs: **{len(nd)}**")
    non = [o for o in opps if not o["actionable"]]
    if non:
        st.caption("Non-actionable chatter (kept as evidence, not ranked): " + "; ".join(f"{x['label']} ({x['tickets']})" for x in non))

# ================= PAGE 3 =================
elif page == "Opportunity Economics":
    st.title("Opportunity Economics")
    signal_panel(ss.get("econ_pick"))
    st.markdown("**Observed facts** (records, accounts, observed ACV) · **User assumptions** (sidebar) · **Model estimates** "
                "(exposure, value, ROI) · **Unknown** (churn, renewal outcomes - never modeled as fact).")
    t = ranked[["final_rank", "label", "unique_accounts", "accounts_observed", "accounts_imputed", "accounts_assumed", "acv_sum_unique",
                "potential_revenue_exposure", "expected_recoverable_value", "total_investment", "net_opportunity_value", "roi",
                "payback_months", "cost_of_delay_per_month"]]
    st.dataframe(t, width="stretch", hide_index=True, key="econ_table")
    with st.expander("Formulas (every number is reproducible)"):
        st.code("unique ACV   = sum over UNIQUE accounts of max(ACV)          # duplicates never multiply revenue\n"
                "exposure     = unique ACV x clip(exposure_fraction x severity/3, 0, 1)\n"
                "recoverable  = exposure x recovery_probability\n"
                "investment   = effort_weeks x eng_cost + infra + support\n"
                "net value    = recoverable - investment;   ROI = net / investment\n"
                "payback (mo) = investment / (recoverable / 12);  cost of delay = recoverable / 12 per month")
    sel = ranked.head(10)
    c1, c2 = st.columns(2)
    c1.plotly_chart(go.Figure([go.Bar(name="Recoverable value", x=sel.label, y=sel.expected_recoverable_value, marker_color="#6c5ce7"),
                               go.Bar(name="Total investment", x=sel.label, y=sel.total_investment, marker_color="#b2bec3")])
                    .update_layout(barmode="group", title="Value vs investment (USD)"), width="stretch")
    c2.plotly_chart(px.scatter(ranked, x="effort_weeks", y="net_opportunity_value", size="unique_accounts", color="evidence_sufficiency",
                               hover_name="label", title="Effort vs value (size = accounts)"), width="stretch")
    st.subheader("Scenario comparison (bear / base / bull)")
    st.dataframe(scenario_ranks([o for o in opps if o["actionable"] and not o["rejected"]], a), width="stretch")
    st.subheader("Confidence range for a theme")
    pick = st.selectbox("Theme", ranked.cluster_id, format_func=lambda i: by_id[i]["label"], key="econ_pick")
    mc = monte_carlo(by_id[pick], a)
    st.write(f"Model-implied net value: P10 {fmt(mc['p10'])} · P50 {fmt(mc['p50'])} · P90 {fmt(mc['p90'])} · P(net>0) {mc['prob_positive']:.0%} "
             "(conditional on assumed ranges; not an empirical probability).")

# ================= PAGE 4 =================
elif page == "Roadmap Optimizer":
    st.title("Roadmap Optimizer")
    signal_panel()
    fw = st.selectbox("Prioritisation framework", list(FRAMEWORKS), key="fw")
    st.caption(f"Formula: {FRAMEWORKS[fw]}")
    w = Weights()
    if fw == "Custom weighted":
        cols = st.columns(8)
        for i, k in enumerate(["revenue", "reach", "severity", "confidence", "urgency", "alignment", "effort", "risk"]):
            setattr(w, k, cols[i].number_input(k, 0.0, 1.0, float(getattr(w, k)), 0.05, key=f"w_{k}"))
        sg = st.columns(4)
        for i, s in enumerate(["Enterprise", "Mid-Market", "SMB", "Unknown"]):
            w.segment_importance[s] = sg[i].slider(f"{s} importance", 0.0, 2.0, 1.0, 0.1, key=f"seg_{s}")
    view = ranked.assign(framework_score=score(ranked, fw, w))
    view["framework_rank"] = view.framework_score.rank(ascending=False, method="first").astype(int)
    st.dataframe(view.sort_values("framework_rank")[["framework_rank", "final_rank", "label", "framework_score", "net_opportunity_value",
                                                     "effort_weeks", "confidence", "evidence_sufficiency", "override_rationale"]],
                 width="stretch", hide_index=True, key="rank_table")
    with st.expander("Compare all frameworks (rank by framework; high spread = contested priority)"):
        st.dataframe(compare_frameworks(ranked, w), width="stretch", hide_index=True)
    if fw == "Custom weighted":
        st.caption("Weight sensitivity: which weights actually change the top 3 at +/-50%?")
        st.dataframe(weight_sensitivity(ranked, w), width="stretch", hide_index=True)

    st.subheader("Constraints")
    k1, k2, k3, k4 = st.columns(4)
    ss.cap = k1.number_input("Engineering capacity (person-weeks)", 0.0, 1000.0, float(ss.get("cap", cap_default)), 1.0, key="cap_in")
    ss.minconf = k2.slider("Min confidence to fund", 0.0, 1.0, float(ss.get("minconf", 0.0)), 0.05, key="minconf_in")
    budget = k3.number_input("Budget cap USD (0 = none)", 0.0, 1e8, float(ss.get("budget") or 0.0), 10000.0, key="budget_in")
    ss.budget = budget or None
    ss.mandatory = k4.multiselect("Mandatory (compliance / committed)", list(ranked.cluster_id), default=[m for m in ss.get("mandatory", []) if m in by_id],
                                  format_func=lambda i: by_id[i]["label"], key="mand_in")
    d1, d2, d3 = st.columns([2, 2, 1])
    child = d1.selectbox("Dependency: this item...", list(ranked.cluster_id), format_func=lambda i: by_id[i]["label"], key="dep_child")
    parent = d2.selectbox("...requires", list(ranked.cluster_id), index=min(1, len(ranked) - 1), format_func=lambda i: by_id[i]["label"], key="dep_par")
    if d3.button("Add dependency", key="b_dep") and child != parent:
        ss.deps = ss.get("deps", []) + [(int(child), int(parent))]
        store.log("dependency", f"{by_id[child]['label']} requires {by_id[parent]['label']}")
    if ss.get("deps"):
        st.caption("Dependencies: " + "; ".join(f"{by_id[c]['label']} → {by_id[p]['label']}" for c, p in ss.deps if c in by_id and p in by_id))
    cons = Constraints(capacity_weeks=ss.cap, min_confidence=ss.minconf, mandatory=ss.mandatory, dependencies=ss.get("deps", []), budget_usd=ss.budget)
    plan = optimise(ranked, cons)
    if plan.status != "optimal":
        st.error(f"No feasible roadmap: {plan.infeasible_reason}")
    else:
        alt = alternative(ranked, cons, plan)
        g = greedy_baseline(ranked, cons)
        m = st.columns(5)
        m[0].metric("Items funded", len(plan.selected)); m[1].metric("Capacity used", f"{plan.utilization:.0%}")
        m[2].metric("Optimised value", fmt(plan.total_value)); m[3].metric("Sort-and-fill value", fmt(g), help="Ranked-list baseline")
        m[4].metric("Alternative plan value", fmt(alt.total_value))
        st.caption(f"Binding constraints: {', '.join(plan.binding) or 'none'}")
        a1, a2 = st.columns(2)
        a1.markdown("**Recommended roadmap**")
        a1.dataframe(plan.selected[["label", "effort_weeks", "net_opportunity_value", "confidence"]], width="stretch", hide_index=True, key="plan_table")
        a2.markdown("**Alternative roadmap** (without the top-value item)")
        a2.dataframe(alt.selected[["label", "effort_weeks", "net_opportunity_value"]], width="stretch", hide_index=True)
        st.markdown("**Deferred & opportunity cost**")
        st.dataframe(plan.deferred[["label", "effort_weeks", "net_opportunity_value", "cost_of_delay_per_month", "deferral_reason"]],
                     width="stretch", hide_index=True)
        rm = generate_roadmap_md(plan.selected, plan.deferred, cons.capacity_weeks, plan.effort_used, plan.binding)
        st.download_button("Export roadmap (.md)", rm, "roadmap.md", key="dl_rm")
        st.download_button("Export roadmap (.csv)", safe_csv(plan.selected), "roadmap.csv", "text/csv", key="dl_rmcsv")

    st.subheader("Override a priority (rationale required)")
    oc = st.selectbox("Theme to move", ranked.cluster_id, format_func=lambda i: by_id[i]["label"], key="ov_theme")
    pos = st.number_input("New rank", 1, len(ranked), 1, key="ov_rank")
    why = st.text_input("Rationale (required)", key="ov_why")
    o1, o2 = st.columns(2)
    if o1.button("Apply override", key="b_override") and why.strip():
        ov.rank[oc] = {"rank": int(pos), "rationale": why.strip()[:300]}
        store.save_overrides(ov); store.log("rank_override", f"{by_id[oc]['label']} -> #{pos}: {why.strip()[:300]}"); st.rerun()
    if o2.button("Remove override for this theme", key="b_unov") and oc in ov.rank:
        ov.rank.pop(oc); store.save_overrides(ov); store.log("rank_override_removed", by_id[oc]["label"]); st.rerun()

# ================= PAGE 5 =================
elif page == "Scenario Lab":
    st.title("Scenario Lab")
    signal_panel()
    st.caption("Simulations show what the model implies under assumed ranges. They do not add precision the data doesn't have.")
    s1, s2, s3 = st.columns(3)
    cap_change = s1.slider("Engineering capacity change", -60, 60, 0, 5, format="%d%%", key="sc_cap")
    rec_change = s2.slider("Recovery probability change", -60, 60, 0, 5, format="%d%%", key="sc_rec")
    seed = s3.number_input("Simulation seed (reproducible)", 0, 10_000, 7, key="sc_seed")
    from dataclasses import replace as _r

    a2 = _r(a, recovery_probability=min(1.0, a.recovery_probability * (1 + rec_change / 100)))
    r2 = rank_opportunities(opps, a2, ov)
    c2 = _r(cons, capacity_weeks=max(0.0, cons.capacity_weeks * (1 + cap_change / 100)))
    p2 = optimise(r2, c2)
    m = st.columns(4)
    m[0].metric("Capacity", f"{c2.capacity_weeks:.0f} pw", f"{cap_change:+d}%")
    m[1].metric("Funded items", len(p2.selected), len(p2.selected) - len(plan.selected))
    m[2].metric("Roadmap value", fmt(p2.total_value), fmt(p2.total_value - plan.total_value))
    m[3].metric("#1 theme", r2.iloc[0].label[:28], "changed" if r2.iloc[0].cluster_id != ranked.iloc[0].cluster_id else "unchanged", delta_color="off")
    st.dataframe(scenario_ranks([o for o in opps if o["actionable"] and not o["rejected"]], a2), width="stretch")
    top = r2.iloc[0]
    st.subheader(f"Sensitivity for #1: {top.label}")
    tn = tornado(by_id[int(top.cluster_id)], a2)
    st.plotly_chart(go.Figure([go.Bar(y=tn.driver, x=tn.delta_low, orientation="h", name="-30%", marker_color="#e17055"),
                               go.Bar(y=tn.driver, x=tn.delta_high, orientation="h", name="+30%", marker_color="#00b894")])
                    .update_layout(barmode="relative", title="Tornado: change in net value (USD)"), width="stretch")
    import numpy as np

    mc = monte_carlo(by_id[int(top.cluster_id)], a2, seed=int(seed))
    st.write(f"Monte Carlo (n={mc['n']}, seed {seed}): P10 {fmt(mc['p10'])} · P50 {fmt(mc['p50'])} · P90 {fmt(mc['p90'])} · "
             f"P(net>0) {mc['prob_positive']:.0%}")
    thr_val = st.number_input("Decision threshold: fund if P50 net value exceeds (USD)", 0.0, 1e8, 50000.0, 10000.0, key="sc_thr")
    st.markdown(f"**Decision at threshold:** {'FUND' if mc['p50'] > thr_val else 'DO NOT FUND YET'} "
                f"(P50 {fmt(mc['p50'])} vs threshold {fmt(thr_val)})")
    rng = np.random.default_rng(int(seed))
    p = a2.recovery_probability
    sims = (by_id[int(top.cluster_id)]["acv_sum_unique"] * top.exposure_fraction * rng.triangular(0.5 * p, p, min(1.0, 1.5 * p) or 1e-9, 3000)
            - top.total_investment * rng.triangular(0.8, 1.0, 1.6, 3000))
    st.plotly_chart(px.histogram(x=sims, nbins=60, title="Uncertainty: simulated net value (USD)", labels={"x": "net value"})
                    .add_vline(x=thr_val, line_dash="dash"), width="stretch")

# ================= PAGE 6 =================
else:
    st.title("Executive Decision Memo")
    signal_panel()
    memo = generate_memo(ranked, a, portfolio_total([o for o in opps if o["actionable"]], a))
    if not is_synth:
        memo = memo.replace("synthetic data unless replaced", f"source: {spec.name if spec else 'user upload'}")
    st.markdown(memo)
    pick = st.selectbox("Document focus (theme)", ranked.label, key="doc_pick")
    row = ranked[ranked.label == pick].iloc[0]
    o = by_id[int(row.cluster_id)]
    prd = generate_prd(row, o, a, monte_carlo(o, a))
    docs = {
        "Executive memo": memo, "PRD": prd, "Opportunity assessment": generate_opportunity_assessment(row, o, a),
        "Evidence appendix": generate_evidence_appendix(o, clustered),
        "Roadmap recommendation": generate_roadmap_md(plan.selected, plan.deferred, cons.capacity_weeks, plan.effort_used, plan.binding),
        "Experiment plan": generate_experiment_plan(ranked), "Decision log": generate_decision_log(store.events()),
    }
    for i, (name, md) in enumerate(docs.items()):
        with st.expander(name, expanded=False):
            st.markdown(md)
        c1, c2, _ = st.columns([1, 1, 4])
        fname = name.lower().replace(" ", "_")
        if c1.download_button(f"{name} (.md)", md, f"{fname}.md", key=f"dlmd_{i}"):
            store.log("export", f"{name} .md")
        c2.download_button(f"{name} (.docx)", markdown_to_docx(md), f"{fname}.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=f"dldocx_{i}")
    full = "\n\n---\n\n".join(docs.values())
    st.download_button("Download full decision pack (.docx)", markdown_to_docx(full), "signalforge_decision_pack.docx", key="dl_pack")
    st.success("Decision-ready. Evidence included.")
    st.subheader("Decision audit trail")
    ev = pd.DataFrame(store.events(), columns=["timestamp (UTC)", "event", "detail"])
    st.dataframe(ev, width="stretch", hide_index=True, key="audit_table")
    st.download_button("Export audit trail (.csv)", safe_csv(ev), "audit_trail.csv", "text/csv", key="dl_audit")

st.sidebar.markdown("---")
if spec:
    with st.sidebar.expander("Data provenance"):
        st.markdown(f"**{spec.name}**  \nLicense: {spec.license}  \nSource: {spec.url}  \n{spec.provenance}  \n*Limits:* {spec.limitations}")
st.sidebar.caption("Session-isolated: your data lives only in this browser session's memory and is discarded when it ends.")
