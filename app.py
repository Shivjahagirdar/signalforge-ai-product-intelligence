"""SignalForge AI - Streamlit decision-support app. Domain logic lives in src/signalforge."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from signalforge.audit import AuditStore, Overrides
from signalforge.economics import Assumptions, monte_carlo, tornado
from signalforge.exports import generate_memo, generate_prd
from signalforge.ingestion import REQUIRED, IngestionError, load_csv, safe_csv, validate
from signalforge.intelligence import cluster_feedback, near_duplicates, suggest_merges
from signalforge.pipeline import build_opportunities, explain_outranks, portfolio_total, rank_opportunities, scenario_ranks
from signalforge.synthetic import generate

st.set_page_config(page_title="SignalForge AI", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.5rem;max-width:1300px}
[data-testid=stMetricValue]{font-size:1.6rem}
.sf-badge{background:#fff3cd;color:#664d03;padding:.2rem .6rem;border-radius:6px;font-size:.8rem}
</style>""", unsafe_allow_html=True)

if "store" not in st.session_state:  # per-visitor, in-memory: nothing is shared between sessions or written to disk
    st.session_state.store = AuditStore(":memory:")
store: AuditStore = st.session_state.store


@st.cache_data(show_spinner="Clustering feedback...")
def run_clustering(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    return cluster_feedback(df, thr)


# ---------------- Sidebar: data + assumptions ----------------
st.sidebar.title("SignalForge AI")
src = st.sidebar.radio("Data source", ["Synthetic demo (1,000 records)", "Upload CSV"])
st.sidebar.download_button("Download CSV template", ",".join(REQUIRED) + "\n", "signalforge_template.csv", "text/csv")
st.sidebar.download_button("Download demo dataset", safe_csv(generate().drop(columns=["ground_truth_issue"])),
                           "signalforge_demo.csv", "text/csv")
ingest = None
try:
    if src.startswith("Synthetic"):
        ingest = validate(generate())
    else:
        up = st.sidebar.file_uploader("Feedback CSV", type=["csv"])
        if up is None:
            st.info("Use the template download in the sidebar. Upload a CSV with columns: ticket_id, account_id, customer_segment, annual_contract_value, "
                    "product_area, feedback_text, severity, frequency, created_at, status, estimated_engineering_effort. "
                    "Do not upload data containing PII you are not authorised to process.")
            st.stop()
        ingest = load_csv(up.getvalue(), up.name)
except IngestionError as e:
    st.error(str(e))
    st.stop()

thr = st.sidebar.slider("Cluster distance threshold", 0.5, 0.98, 0.9, 0.01,
                        help="Higher = fewer, broader clusters.")
st.sidebar.subheader("Financial assumptions")
a = Assumptions(
    base_exposure_fraction=st.sidebar.slider("Exposure fraction of ACV", 0.0, 1.0, 0.30, 0.05),
    recovery_probability=st.sidebar.slider("Recovery probability", 0.0, 1.0, 0.40, 0.05),
    eng_cost_per_person_week=st.sidebar.number_input("Eng cost / person-week (USD)", 0.0, 50000.0, 4000.0, 500.0),
    infra_cost_annual=st.sidebar.number_input("Incremental infra / yr (USD)", 0.0, 1e6, 2000.0, 500.0),
    support_cost_annual=st.sidebar.number_input("Incremental support / yr (USD)", 0.0, 1e6, 3000.0, 500.0),
    impute_missing_acv=st.sidebar.checkbox("Impute missing ACV (segment median)", True),
).validated()
if st.sidebar.button("Reset local data & overrides"):
    store.reset()
    st.rerun()

clustered = run_clustering(ingest.df, thr)
sig = (src, len(ingest.df), thr, str(ingest.df["ticket_id"].iloc[0]))
if st.session_state.get("sig") != sig:  # cluster ids change with data/threshold, so stale overrides must not carry over
    if st.session_state.get("sig") is not None:
        store.save_overrides(Overrides()); store.log("overrides_cleared", "Data source or threshold changed")
    st.session_state["sig"] = sig
ov: Overrides = store.load_overrides()
opps = build_opportunities(clustered, a, ov)
ranked = rank_opportunities(opps, a, ov)
if ranked.empty:
    st.warning("No opportunities found (need clusters with 3+ tickets).")
    st.stop()
by_id = {o["cluster_id"]: o for o in opps}
if src.startswith("Synthetic"):
    st.markdown('<span class="sf-badge">SYNTHETIC DEMO DATA - not real customers</span>', unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", ["Command Center", "Customer Intelligence", "Opportunity Economics",
                                      "Roadmap Optimizer", "Scenario Lab", "Decision Memo & PRD"])
fmt = lambda v: f"${v:,.0f}"

if page == "Command Center":
    st.header("Executive Command Center")
    c = st.columns(5)
    c[0].metric("Records analyzed", f"{len(ingest.df):,}")
    c[1].metric("Unique accounts", f"{ingest.df.account_id.nunique():,}")
    c[2].metric("Pain themes", len(ranked))
    c[3].metric("Potential exposure (scenario)", fmt(portfolio_total(opps, a)))
    c[4].metric("Evidence coverage", f"{ranked.tickets.sum() / len(ingest.df):.0%}")
    for w in ingest.warnings:
        st.caption(f"Data note: {w}")
    st.subheader("Top 3 recommended investments")
    for r in ranked.head(3).itertuples():
        with st.container(border=True):
            st.markdown(f"**#{r.final_rank} - {r.label}**  ({r.product_area})")
            st.write(f"{r.unique_accounts} unique accounts - exposure {fmt(r.potential_revenue_exposure)} - "
                     f"net value {fmt(r.net_opportunity_value)} - {r.effort_weeks:.1f} person-weeks - evidence {r.confidence:.2f}")
    if len(ranked) > 1:
        st.info(explain_outranks(ranked.iloc[0], ranked.iloc[1]))
    st.caption("Exposure is a scenario estimate, not verified churn or revenue loss.")

elif page == "Customer Intelligence":
    st.header("Customer Intelligence")
    q = st.text_input("Search feedback")
    view = clustered if not q else clustered[clustered.feedback_text.str.contains(q, case=False, regex=False)]
    st.dataframe(view[["ticket_id", "account_id", "customer_segment", "severity", "cluster_label", "feedback_text"]],
                 use_container_width=True, height=280)
    st.download_button("Download processed CSV", safe_csv(clustered), "signalforge_processed.csv", "text/csv")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(clustered, x="severity", color="customer_segment", title="Severity by segment"),
                    use_container_width=True)
    c2.plotly_chart(px.bar(ranked, x="label", y="tickets", title="Tickets per theme"), use_container_width=True)
    nd = near_duplicates(ingest.df)
    st.write(f"Exact duplicates removed at ingestion: **{ingest.duplicates_removed}** - near-duplicate same-account pairs flagged: **{len(nd)}**")
    st.subheader("Suggested merges (clusters that may be paraphrases of one problem)")
    sm = suggest_merges(clustered)
    if sm.empty:
        st.caption("No similar cluster pairs found.")
    else:
        lab = clustered.drop_duplicates("cluster_id").set_index("cluster_id")["cluster_label"]
        st.dataframe(sm.assign(label_a=sm.cluster_a.map(lab), label_b=sm.cluster_b.map(lab)), use_container_width=True)
    st.subheader("Evidence drill-down & human corrections")
    cid = st.selectbox("Theme", ranked.cluster_id, format_func=lambda i: f"{i}: {by_id[i]['label']}")
    st.dataframe(clustered[clustered.ticket_id.isin(by_id[cid]["ticket_ids"])][["ticket_id", "account_id", "feedback_text"]],
                 use_container_width=True, height=220)
    n1, n2 = st.columns(2)
    new = n1.text_input("Rename theme")
    if n1.button("Apply rename") and new:
        ov.rename[cid] = new
        store.save_overrides(ov); store.log("rename", f"cluster {cid} -> {new}"); st.rerun()
    other = n2.selectbox("Merge into this theme", [i for i in ranked.cluster_id if i != cid],
                         format_func=lambda i: f"{i}: {by_id[i]['label']}")
    if n2.button("Merge selected into theme above"):
        ov.merge.append([cid, other])
        store.save_overrides(ov); store.log("merge", f"{other} merged into {cid}"); st.rerun()
    bad = st.multiselect("Mark evidence irrelevant", by_id[cid]["ticket_ids"])
    if st.button("Exclude selected tickets") and bad:
        ov.irrelevant += bad
        store.save_overrides(ov); store.log("irrelevant", f"{len(bad)} tickets excluded from cluster {cid}"); st.rerun()

elif page == "Opportunity Economics":
    st.header("Opportunity Economics")
    t = ranked[["final_rank", "label", "unique_accounts", "acv_sum_unique", "potential_revenue_exposure",
                "expected_recoverable_value", "total_investment", "net_opportunity_value", "roi", "payback_months",
                "cost_of_delay_per_month"]]
    st.dataframe(t, use_container_width=True)
    st.caption("USD. ACV counted once per unique account. Net = recoverable - (eng + operating). Cost of delay = USD/month of recoverable value forgone.")
    sel = ranked.head(10)
    st.plotly_chart(go.Figure([go.Bar(name="Recoverable value", x=sel.label, y=sel.expected_recoverable_value),
                               go.Bar(name="Total investment", x=sel.label, y=sel.total_investment)]).update_layout(
        barmode="group", title="Value vs investment (USD)"), use_container_width=True)
    st.plotly_chart(px.scatter(ranked, x="effort_weeks", y="net_opportunity_value", size="unique_accounts", text="label",
                               title="Engineering effort vs business value"), use_container_width=True)

elif page == "Roadmap Optimizer":
    st.header("Roadmap Optimizer")
    st.dataframe(ranked[["final_rank", "label", "priority", "model_rank", "rice_rank", "baseline_rank",
                         "net_opportunity_value", "effort_weeks", "confidence", "override_rationale"]], use_container_width=True)
    st.caption("model_rank = risk-adjusted priority; rice_rank = RICE; baseline_rank = severity x frequency (rules-based).")
    c1, c2 = st.columns(2)
    hi = c1.selectbox("Feature A", ranked.label, 0)
    lo = c2.selectbox("Feature B", ranked.label, min(1, len(ranked) - 1))
    if hi != lo:
        st.write(explain_outranks(ranked[ranked.label == hi].iloc[0], ranked[ranked.label == lo].iloc[0]))
    st.subheader("Override a ranking (with rationale)")
    oc = st.selectbox("Theme to move", ranked.cluster_id, format_func=lambda i: by_id[i]["label"])
    pos = st.number_input("New rank", 1, len(ranked), 1)
    why = st.text_input("Rationale (required)")
    if st.button("Apply override") and why:
        ov.rank[oc] = {"rank": int(pos), "rationale": why}
        store.save_overrides(ov); store.log("rank_override", f"{by_id[oc]['label']} -> #{pos}: {why}"); st.rerun()

elif page == "Scenario Lab":
    st.header("Scenario Lab")
    st.dataframe(scenario_ranks(opps, a), use_container_width=True)
    top = ranked.iloc[0]
    st.subheader(f"Sensitivity for #1: {top.label}")
    tn = tornado(by_id[int(top.cluster_id)], a)
    st.plotly_chart(go.Figure([go.Bar(y=tn.driver, x=tn.delta_low, orientation="h", name="-30%"),
                               go.Bar(y=tn.driver, x=tn.delta_high, orientation="h", name="+30%")]).update_layout(
        barmode="relative", title="Tornado: change in net value (USD)"), use_container_width=True)
    mc = monte_carlo(by_id[int(top.cluster_id)], a)
    st.write(f"Model-implied net value (conditional on assumptions, not an empirical probability): "
             f"P10 {fmt(mc['p10'])} - P50 {fmt(mc['p50'])} - P90 {fmt(mc['p90'])} - P(net>0) {mc['prob_positive']:.0%}")

else:
    st.header("Decision Memo & PRD")
    memo = generate_memo(ranked, a, portfolio_total(opps, a))
    st.markdown(memo)
    st.download_button("Download executive summary (.md)", memo, "executive_summary.md")
    pick = st.selectbox("PRD for", ranked.label)
    row = ranked[ranked.label == pick].iloc[0]
    prd = generate_prd(row, by_id[int(row.cluster_id)], a, monte_carlo(by_id[int(row.cluster_id)], a))
    with st.expander("Preview PRD"):
        st.markdown(prd)
    st.download_button("Download PRD (.md)", prd, "PRD.md")
    st.subheader("Decision audit trail")
    st.dataframe(pd.DataFrame(store.events(), columns=["timestamp (UTC)", "event", "detail"]), use_container_width=True)
