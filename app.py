"""SignalForge AI - Product Intelligence & Decision OS (Streamlit UI). Domain logic lives in src/signalforge."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from signalforge import evidence_quality as EQ
from signalforge import lifecycle as LC
from signalforge import telemetry as TM
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
    markdown_to_pdf,
)
from signalforge.github_sync import sync_repo
from signalforge.ingestion import MAX_BYTES, REQUIRED, IngestionError, safe_csv, validate
from signalforge.intelligence import SEMANTIC_THRESHOLD, cluster_feedback, near_duplicates, suggest_merges
from signalforge.metrics import session_metrics
from signalforge.optimizer import Constraints, alternative, greedy_baseline, optimise
from signalforge.persistence import restore as restore_snapshot
from signalforge.persistence import snapshot
from signalforge.pipeline import build_opportunities, explain_outranks, portfolio_total, rank_opportunities, scenario_ranks
from signalforge.prioritization import FRAMEWORKS, Weights, compare_frameworks, score, weight_sensitivity
from signalforge.store import Store, StoreError, new_workspace_id, valid_workspace_id
from signalforge.synthetic import generate

st.set_page_config(page_title="SignalForge AI", page_icon="◆", layout="wide")
POP = ["#0b2545", "#b08d57", "#3e6d9c", "#8da9c4", "#5c7a6b", "#a4508b", "#c0392b", "#6b7280"]
pio.templates["sf"] = go.layout.Template(layout=dict(
    colorway=POP, font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", color="#13233a", size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    title=dict(font=dict(family="Newsreader, Georgia, serif", size=18, color="#0b2545")),
    xaxis=dict(gridcolor="#e8e4da", zerolinecolor="#d8d2c4", linecolor="#d8d2c4"),
    yaxis=dict(gridcolor="#e8e4da", zerolinecolor="#d8d2c4", linecolor="#d8d2c4"),
    legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=10, r=10, t=52, b=10)))
pio.templates.default = "plotly_white+sf"
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
:root{--navy:#0b2545;--ink:#13233a;--muted:#5f6b7a;--line:#e3ded3;--paper:#faf8f3;--gold:#b08d57;--card:#ffffff}
html,body,.stApp,.stMarkdown,p,li,label,button,input,textarea{font-family:Inter,-apple-system,"Segoe UI",Roboto,sans-serif}
.stApp{background:var(--paper)}
header[data-testid=stHeader]{background:transparent}
.block-container{padding-top:3.2rem;max-width:1240px}
.stApp p,.stApp li{color:var(--ink);line-height:1.6}
.stApp h1{font-family:Newsreader,Georgia,serif;font-weight:600;font-size:2.55rem;letter-spacing:-.01em;color:var(--navy);margin-bottom:.2rem}
.stApp h2,.stApp h3{font-family:Newsreader,Georgia,serif;font-weight:600;color:var(--navy);letter-spacing:0}
.stApp h3{font-size:1.4rem;border-top:1px solid var(--line);padding-top:1.1rem;margin-top:1.2rem}
[data-testid=stSidebar]{background:var(--navy);border-right:0}
[data-testid=stSidebar] *{color:#e6ebf2}
[data-testid=stSidebar] p,[data-testid=stSidebar] label,[data-testid=stSidebar] span,[data-testid=stSidebar] small,[data-testid=stSidebar] li{color:#e6ebf2}
[data-testid=stSidebar] [data-testid=stRadioOption]>div>div:first-child:not([data-testid=stMarkdownContainer]){display:none}
[data-testid=stSidebar] [data-testid=stRadioOption][data-selected=true]{background:rgba(255,255,255,.1);border-left-color:var(--gold)}
[data-testid=stSidebar] [data-testid=stRadioOption][data-selected=true] p{font-weight:600;color:#fff}
[data-testid=stSidebar] [data-testid=stExpander]{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.14)}
[data-testid=stSidebar] [data-testid=stExpander] summary:hover{background:rgba(255,255,255,.05)}
[data-testid=stSidebar] [data-testid=stExpander] details{background:transparent}
[data-testid=stSidebar] .stButton button,[data-testid=stSidebar] .stDownloadButton button{background:transparent;border:1px solid rgba(255,255,255,.3)}
[data-testid=stSidebar] .stButton button:hover,[data-testid=stSidebar] .stDownloadButton button:hover{border-color:var(--gold);background:rgba(255,255,255,.06)}
[data-testid=stSidebar] h5{font-size:.7rem!important;letter-spacing:.14em;text-transform:uppercase;color:#9fb0c6!important;font-weight:600}
[data-testid=stSidebar] hr{border-color:rgba(255,255,255,.12)}
[data-testid=stSidebar] [data-baseweb=select]>div,[data-testid=stSidebar] input,[data-testid=stSidebar] textarea{background:#12305a!important;border-color:#24446f!important}
[data-testid=stSidebar] [data-testid=stRadio] label{border-radius:6px;padding:.45rem .7rem;margin-bottom:.1rem;width:100%;border-left:3px solid transparent}
[data-testid=stSidebar] [data-testid=stRadio] label:hover{background:rgba(255,255,255,.06)}
[data-testid=stSidebar] [data-testid=stRadio] label:has(input:checked){background:rgba(255,255,255,.1);border-left-color:var(--gold)}
[data-testid=stSidebar] [data-testid=stRadio] label:has(input:checked) p{font-weight:600;color:#fff}
[data-testid=stSidebar] [data-testid=stRadio] label>div:first-child{display:none}
.sf-hero{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:2px solid var(--navy);padding:0 0 .7rem;margin:0 0 1.1rem}
.sf-logo{display:none}
.sf-brand{font-family:Newsreader,Georgia,serif;font-size:1.35rem;font-weight:600;color:var(--navy)}
.sf-brand:before{content:"◆ ";color:var(--gold)}
.sf-tag{font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);text-align:right}
.sf-steps{display:flex;gap:0;margin:.1rem 0 1.4rem;border:1px solid var(--line);background:var(--card);border-radius:4px;overflow:hidden}
.sf-step{flex:1;padding:.55rem .7rem;font-size:.72rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#8a93a0;
 border-right:1px solid var(--line);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sf-step:last-child{border-right:0}
.sf-step.done{color:var(--navy)}
.sf-step.on,.sf-step.now{background:var(--navy);color:#fff;box-shadow:inset 0 -3px 0 var(--gold)}
.sf-blurb{font-family:Newsreader,Georgia,serif;font-size:1.25rem;color:var(--muted);margin:0 0 1.4rem;max-width:62rem;line-height:1.45}
[data-testid=stMetric]{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:1rem 1.1rem;border-top:3px solid var(--navy)}
[data-testid=stMetricValue]{font-family:Newsreader,Georgia,serif;font-size:2rem;font-weight:600;color:var(--navy)}
[data-testid=stMetricLabel] p{color:var(--muted);font-size:.72rem!important;letter-spacing:.1em;text-transform:uppercase;font-weight:600}
.stButton button,.stDownloadButton button{border-radius:4px;font-weight:600;font-size:.88rem;background:var(--navy);color:#fff;border:1px solid var(--navy);box-shadow:none}
.stButton button p,.stDownloadButton button p{color:#fff}
.stButton button:hover,.stDownloadButton button:hover{background:#163a66;border-color:#163a66;color:#fff}
[class*="st-key-q_"] button{background:var(--card);border:1px solid var(--line)}
[class*="st-key-q_"] button p{color:var(--navy)}
[class*="st-key-q_"] button:hover{border-color:var(--gold);background:var(--card)}
[class*="st-key-next_"] button{background:transparent;border:1px solid var(--navy)}
[class*="st-key-next_"] button p{color:var(--navy)}
[class*="st-key-next_"] button:hover{background:var(--navy)}
[class*="st-key-next_"] button:hover p{color:#fff}
[data-testid=stExpander]{background:var(--card);border-radius:4px;border:1px solid var(--line)}
div[data-testid=stVerticalBlockBorderWrapper]{border-radius:4px}
[data-testid=stChatMessage]{border-radius:4px;background:var(--card);border:1px solid var(--line)}
[data-testid=stDataFrame]{border:1px solid var(--line);border-radius:4px}
.stTabs [data-baseweb=tab]{font-weight:600;color:var(--muted)}
.stTabs [aria-selected=true]{color:var(--navy)}
.stTabs [data-baseweb=tab-highlight]{background:var(--gold)}
.sf-pill{display:inline-block;padding:.15rem .55rem;border-radius:3px;font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-right:.35rem;border:1px solid}
.sf-syn{background:#fbf5e9;color:#80612d;border-color:#e6d5b0}.sf-real{background:#eef5f0;color:#2f5d43;border-color:#c5dccd}
.sf-warn{background:#fbefed;color:#8f2d22;border-color:#ebc6c0}.sf-sim{background:#eef2f8;color:#2d4a72;border-color:#c9d5e6}
.sf-banner{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--gold);padding:.95rem 1.2rem;border-radius:4px;margin:.4rem 0 1.2rem}
.sf-banner b{color:var(--navy)}
.sf-muted{color:var(--muted);font-size:.85rem}
.sf-card{border-radius:4px;padding:1rem 1.2rem;margin:.3rem 0;border:1px solid var(--line);background:var(--card)}
.sf-good{border-left:3px solid #2f5d43}.sf-mid{border-left:3px solid var(--gold)}.sf-bad{border-left:3px solid #8f2d22}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="sf-hero"><div class="sf-logo">◆</div><div class="sf-brand">SignalForge AI</div>'
            '<div class="sf-tag">Product Decision Intelligence</div></div>', unsafe_allow_html=True)
fmt = lambda v: "n/a" if v is None or pd.isna(v) else f"${v:,.0f}"  # noqa: E731


def short(v: float) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"${v / div:.1f}{suf}"
    return f"${v:,.0f}"

# ---------------- per-visitor state (in-memory; never shared, never written to disk) ----------------
ss = st.session_state
if "store" not in ss:
    ss.store = AuditStore(":memory:")
    ss.chat = []
    ss.onboarded = False
    ss.t0 = time.time()
store: AuditStore = ss.store


@st.cache_resource
def durable_store() -> Store:
    return Store()  # SIGNALFORGE_DB or data/signalforge.db: survives reloads/sessions, not a platform redeploy


DB = durable_store()
if "ws" not in ss:
    qp = st.query_params.get("ws", "")
    ss.ws = qp if valid_workspace_id(qp) and DB.workspace_exists(qp) else DB.ensure_workspace(new_workspace_id())
WS: str = ss.ws


@st.cache_data(show_spinner="Finding themes in your feedback...", max_entries=8)
def run_clustering(df: pd.DataFrame, thr: float, semantic: bool) -> pd.DataFrame:
    return cluster_feedback(df, thr, semantic=semantic)


@st.cache_data(show_spinner="Fetching public dataset from its original host...", ttl=3600, max_entries=6)
def fetch_public(key: str, repo: str | None) -> pd.DataFrame:
    return load_public(key, repo)  # cached per (dataset, repo): public data only, never user uploads


def _go(target: str) -> None:
    ss.page = target


def _on_semantic() -> None:
    ss.thr = SEMANTIC_THRESHOLD if ss.get("semantic") else 0.9  # distance scale differs between backends


def _on_restore() -> None:
    f = ss.get("restore_file")
    if f is None:
        return
    try:
        snap = restore_snapshot(f.getvalue())
    except ValueError as e:
        ss.restore_error = str(e)
        return
    for k, v in snap["widgets"].items():  # set widget state BEFORE widgets are created on the rerun
        ss[k] = v
    ss.pending_restore = snap


# ---------------- sidebar: data ----------------
PAGES = ["Executive Dashboard", "Customer Intelligence", "Opportunity Economics", "Roadmap Optimizer", "Decision Loop", "Scenario Lab", "Decision Memo & PRD"]
ICON = {p: f"{i + 1:02d}" for i, p in enumerate(PAGES)}
st.sidebar.markdown("##### Navigation")
page = st.sidebar.radio("Navigate", PAGES, key="page", format_func=lambda n: f"{ICON[n]}   {n}", label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("##### Data source")
SOURCES = {"Synthetic demo (1,000 records)": "synthetic", "Public: F-Droid app reviews": "fdroid_reviews",
           "Public: Bitext support requests": "bitext_support", "Public: GitHub issues (live)": "github_issues", "Upload file": "upload"}
src_label = st.sidebar.selectbox("Dataset", list(SOURCES), key="dataset")
src = SOURCES[src_label]
st.sidebar.download_button("⬇ CSV template", ",".join(REQUIRED) + "\n", "signalforge_template.csv", "text/csv")

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
    elif src == "github_issues":
        repo = st.sidebar.text_input("Public repo (owner/name)", "streamlit/streamlit", key="gh_repo").strip()
        key_src = f"github:{repo}"
        have = DB.feedback(WS, key_src) if repo else []
        cs1, cs2 = st.sidebar.columns(2)
        do_sync = cs1.button("Sync now", key="gh_sync", help="Incremental: only issues updated since the last sync")
        full = cs2.button("Full resync", key="gh_full")
        if (do_sync or full or not have) and repo:
            with st.spinner("Syncing GitHub issues (paginated, retried, rate-limit aware)..."):
                rep = sync_repo(DB, WS, repo, full=full)
            ss.gh_report = rep
            have = DB.feedback(WS, key_src)
        info = DB.get_sync(WS, key_src) if repo else None
        if info:
            st.sidebar.caption(f"Last sync {info['last_run']} UTC - {info['last_status']}. Cursor: {info['cursor'] or 'none'}. "
                               f"{len(have)} issues stored in your workspace.")
        for w_ in getattr(ss.get("gh_report"), "warnings", []):
            st.sidebar.caption(f"Note: {w_}")
        if not have:
            raise IngestionError("No issues found for that repository yet.")
        ingest = validate(pd.DataFrame(have).drop(columns=["updated_at", "source_url"], errors="ignore"))
    else:
        ingest = validate(fetch_public(src, None))
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
st.sidebar.markdown("##### Analysis settings")
semantic = st.sidebar.toggle("Semantic matching (beta)", False, key="semantic", on_change=_on_semantic,
                             help="Groups paraphrases better (held-out ARI 0.30 → 0.76 in CI) but loosens tightly-templated data "
                                  "(synthetic ARI 0.999 → 0.37 at the same settings). Downloads a ~30 MB open model once.")
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
with st.sidebar.expander("Save / load analysis"):
    st.caption("No server-side storage: download a snapshot (settings, assumptions, overrides, constraints, audit) and "
               "load it later to continue. Raw data is never included.")
    _snap_widgets = {k: ss.get(k) for k in ("dataset", "thr", "exp", "rec", "eng", "infra", "sup", "impute", "assumed", "semantic")}
    _cons = {"cap": ss.get("cap"), "minconf": ss.get("minconf", 0.0), "budget": ss.get("budget"),
             "mandatory": [int(x) for x in ss.get("mandatory", [])], "deps": [list(map(int, d)) for d in ss.get("deps", [])]}
    st.download_button("Save analysis (.json)", snapshot(_snap_widgets, store.load_overrides(), _cons, store.events(),
                                                       [str(x) if not isinstance(x, (int, float)) else x for x in ss.get("sig", ())]),
                       "signalforge_analysis.json", "application/json", key="save_analysis")
    st.file_uploader("Load saved analysis", type=["json"], key="restore_file", on_change=_on_restore)
if st.sidebar.button("↺ Reset my session", key="reset", help="Clears data, overrides and the audit trail for this session."):
    store.reset(); ss.chat = []
    st.rerun()

with st.sidebar.expander("Workspace"):
    st.code(WS, language=None)
    st.caption("Decisions, tickets, releases, outcomes and GitHub syncs are saved to this workspace. Bookmark "
               f"`?ws={WS}` or paste the key below to resume later. Hosted demo storage resets on redeploy.")
    other = st.text_input("Resume workspace key", key="ws_resume")
    if st.button("Open workspace", key="b_ws_open"):
        if valid_workspace_id(other.strip()) and DB.workspace_exists(other.strip()):
            ss.ws = other.strip(); st.query_params["ws"] = ss.ws; st.rerun()
        else:
            st.error("No workspace with that key.")
st.sidebar.caption("Analysis state is per browser session. Workspace data is isolated by key and enforced in every database query.")

# ---------------- analysis ----------------
clustered = run_clustering(ingest.df, thr, bool(semantic))
if semantic and "model2vec" not in str(clustered.attrs.get("backend")):
    st.sidebar.warning("Semantic model unavailable on this host; using the lexical backend.")
sig = (src, len(ingest.df), thr, bool(semantic), str(ingest.df["ticket_id"].iloc[0]))
if ss.get("sig") != sig:  # cluster ids change with data/threshold: stale overrides must not silently carry over
    if ss.get("sig") is not None:
        store.save_overrides(Overrides()); store.log("overrides_cleared", "Data source or threshold changed")
        for k in ("mandatory", "deps"):  # constraints reference cluster ids of the previous data
            ss.pop(k, None)
        ss.chat = []  # SIGNAL answers were about the previous dataset
    ss["sig"] = sig
if ss.get("pending_restore"):
    snap = ss.pop("pending_restore")
    store.save_overrides(snap["overrides"])
    store.import_events(snap["audit"])
    c = snap["constraints"]
    for k in ("cap", "minconf", "budget", "mandatory", "deps"):
        if c.get(k) is not None:
            ss[k] = c[k]
    store.log("analysis_restored", f"saved analysis loaded ({len(snap['audit'])} audit events)")
    if snap.get("data_signature") and list(snap["data_signature"]) != [str(x) if not isinstance(x, (int, float)) else x for x in sig]:
        st.warning("Saved analysis was made on different data; theme overrides may not line up. Review them before relying on them.")
    else:
        st.toast("Saved analysis restored. Overrides, constraints and audit trail are back.")
if ss.get("restore_error"):
    st.error(f"That file isn't cooperating. {ss.pop('restore_error')}")
ov: Overrides = store.load_overrides()
opps = build_opportunities(clustered, a, ov)
ranked = rank_opportunities(opps, a, ov)
by_id = {o["cluster_id"]: o for o in opps}
dq = data_quality(ingest.df)
quality = {o['cluster_id']: EQ.assess(o, clustered) for o in opps}
if ranked.empty:
    st.warning("This theme needs more evidence before it earns roadmap space: no actionable cluster has 3+ records. "
               "Try a broader cluster setting or more data.")
    st.stop()

if "t_first" not in ss:
    ss.t_first = time.time() - ss.t0  # server-side: session start -> first fully computed analysis
pill = ('<span class="sf-pill sf-syn">SYNTHETIC DEMO - fictional accounts & revenue</span>' if is_synth else
        f'<span class="sf-pill sf-real">REAL PUBLIC DATA - {spec.name}</span>' if spec else '<span class="sf-pill sf-real">YOUR UPLOAD</span>')
if not has_acv:
    pill += '<span class="sf-pill sf-warn">NO REVENUE DATA - $ values use your assumption</span>'
st.markdown(pill, unsafe_allow_html=True)

STEPS = ["Overview", "Evidence", "Economics", "Roadmap", "Decision loop", "Scenarios", "Memo"]
BLURB = {
    "Executive Dashboard": "Where product investment will create the most value this quarter, and the evidence behind each call.",
    "Customer Intelligence": "Customer feedback consolidated into evidence-linked themes, each traceable to its source records and open to human review.",
    "Opportunity Economics": "Value at stake, cost to deliver and expected return for every opportunity, with facts, assumptions and estimates kept distinct.",
    "Roadmap Optimizer": "A capacity-feasible portfolio that maximises value within your constraints, alongside the trade-offs it implies.",
    "Decision Loop": "From approved decision to engineering ticket, release and measured outcome, with forecast compared to actual.",
    "Scenario Lab": "Stress-test the recommendation against shocks and uncertainty before committing capital.",
    "Decision Memo & PRD": "Board-ready materials: executive memo, PRD, evidence appendix and a complete decision audit trail.",
}
cur = PAGES.index(page)
st.markdown('<div class="sf-steps">' + "".join(
    f'<span class="sf-step {"on" if i == cur else "done" if i < cur else ""}">{ICON[PAGES[i]]}&nbsp;&nbsp;{STEPS[i]}</span>'
    for i in range(len(PAGES))) + "</div>", unsafe_allow_html=True)


def next_step() -> None:
    if cur < len(PAGES) - 1:
        st.button(f"Continue to {PAGES[cur + 1]}  →", on_click=_go, args=(PAGES[cur + 1],), key=f"next_{cur}")

# ---------------- roadmap constraints (shared by optimizer, copilot, memo) ----------------
cap_default = float(round(ranked["effort_weeks"].sum() * 0.6, 1)) or 10.0
cons = Constraints(capacity_weeks=ss.get("cap", cap_default), min_confidence=ss.get("minconf", 0.0),
                   mandatory=ss.get("mandatory", []), dependencies=ss.get("deps", []),
                   budget_usd=ss.get("budget") or None)
plan = optimise(ranked, cons)


# ---------------- SIGNAL copilot (global, page-aware, session-scoped) ----------------
def signal_panel(selected: int | None = None) -> None:
    with st.expander("◆ SIGNAL - your AI product strategist", expanded=bool(ss.chat)):
        st.caption("Ask anything about this analysis. Answers are grounded in this session's data and calculations. " +
                   ("LLM narration on (server-side key)." if llm_available() else "Deterministic analytics (no LLM key needed)."))
        quick = ["What should we build next?", "Which three issues threaten the most revenue?", "Why is this cluster ranked first?",
                 "What happens if engineering capacity drops 30%?", "What would a CFO prioritize?", "Which recommendations have weak evidence?",
                 "What are we ignoring?", "Challenge this roadmap.", "What assumptions would reverse this decision?", "Run the decision debate",
                 "How solid is the evidence?", "Did the release work?"]
        cols = st.columns(5)
        clicked = None
        for i, qq in enumerate(quick):
            if cols[i % 5].button(qq, key=f"q_{page}_{i}", width="stretch"):
                clicked = qq
        q = st.text_input("Ask SIGNAL", key=f"ask_{page}", placeholder="e.g. Which feature should we kill?")
        if st.button("Ask", key=f"askbtn_{page}") and q:
            clicked = q
        if clicked:
            decs = DB.decisions(WS)
            stt = CopilotState(ranked, by_id, a, cons, page, selected, "synthetic" if is_synth else "real",
                               loop={"decisions": decs, "outcomes": {d_.id: DB.outcomes(WS, d_.id) for d_ in decs}, "quality": quality})
            rec = answer(stt, clicked)
            text = narrate_with_llm(rec) if llm_available() else None
            ss.chat.insert(0, (clicked, text or rec.to_markdown(), rec.table))
            store.log("copilot", f"[{rec.tool}] {clicked[:120]}")
        for qq, md, table in ss.chat[:3]:
            with st.chat_message("user"):
                st.markdown(qq)
            with st.chat_message("assistant", avatar=":material/insights:"):
                st.markdown(md)
                if table:
                    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)


def insight_banner() -> str:
    r0 = ranked.iloc[0]
    if (ranked.net_opportunity_value <= 0).all():
        return ("<b>SIGNAL:</b> At the current assumptions no theme pays for its engineering cost. Treat this as a pain map, "
                "not a business case - or bring real revenue data.")
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
    st.markdown(f'<div class="sf-blurb">{BLURB["Executive Dashboard"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sf-banner"><b>Operating model.</b> evidence-linked themes → revenue economics → capacity-feasible roadmap → '
                'human-approved decision → ticket + PRD → release → measured outcome. Run it in <b>Decision Loop</b>.</div>', unsafe_allow_html=True)
    signal_panel()
    if not ss.onboarded:
        with st.container(border=True):
            st.markdown("**First time here?** 1) Pick a dataset in the sidebar (synthetic demo or real public data). "
                        "2) Review themes in *Customer Intelligence*. 3) Tune assumptions and see *Economics* recalculate. "
                        "4) Set capacity in *Roadmap Optimizer*. 5) Ask SIGNAL to challenge it. 6) Run the *Decision Loop*: approve → ticket → release → measure. 7) Export the memo.")
            if st.button("Got it", key="onb"):
                ss.onboarded = True; st.rerun()
    st.markdown(f'<div class="sf-banner">{insight_banner()}</div>', unsafe_allow_html=True)
    c = st.columns(6)
    c[0].metric("Records", f"{len(ingest.df):,}", help=f"{ingest.dropped_rows} rejected, {ingest.duplicates_removed} exact duplicates removed")
    c[1].metric("Accounts", f"{ingest.df.account_id.nunique():,}")
    c[2].metric("Themes", len(ranked), help=f"{sum(not o['actionable'] for o in opps)} non-actionable chatter clusters excluded")
    c[3].metric("Exposure*", short(portfolio_total([o for o in opps if o["actionable"]], a)),
                help="Each account counted once at its max exposure fraction. Scenario estimate, not churn.")
    c[4].metric("Plan value", short(plan.total_value), help=f"{len(plan.selected)} items in {cons.capacity_weeks:.0f} person-weeks")
    c[5].metric("Data quality", f"{dq['score']:.0f}/100", help=str({k: f'{v:.0%}' for k, v in dq['components'].items()}))
    for w in ingest.warnings + notes:
        st.caption(f"Data note: {w}")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Recommended executive actions")
        for r in ranked.head(3).itertuples():
            with st.container(border=True):
                badge = {"sufficient": "●", "directional": "◐", "insufficient": "○"}[r.evidence_sufficiency]
                st.markdown(f"**{int(r.final_rank):02d} — {r.label}** · {r.product_area} · {badge} evidence {r.evidence_sufficiency}")
                st.write(f"{r.unique_accounts} accounts · exposure {fmt(r.potential_revenue_exposure)} · net {fmt(r.net_opportunity_value)} · "
                         f"{r.effort_weeks:.1f} pw · confidence {r.confidence:.2f} · trend {r.trend_status}")
        if len(ranked) > 1:
            st.info(explain_outranks(ranked.iloc[0], ranked.iloc[1]))
    with right:
        st.subheader("Top product risks")
        risk = ranked.sort_values("potential_revenue_exposure", ascending=False).head(6)
        st.plotly_chart(px.bar(risk, x="potential_revenue_exposure", y="label", orientation="h", color="avg_severity",
                               color_continuous_scale="Sunsetdark", labels={"potential_revenue_exposure": "Exposure (USD)", "label": ""})
                        .update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis={"autorange": "reversed"}),
                        width="stretch", theme=None)
        em = ranked[ranked.trend_status.isin(["emerging", "rising"])]
        st.subheader("Emerging issues")
        st.write(", ".join(f"{x.label} ({x.trend_status})" for x in em.itertuples()) or "None detected (or not enough dated records).")
    st.caption("Every KPI's calculation is in Opportunity Economics; every theme's evidence is in Customer Intelligence.")

    next_step()

# ================= PAGE 2 =================
elif page == "Customer Intelligence":
    st.title("Customer Intelligence")
    st.markdown(f'<div class="sf-blurb">{BLURB["Customer Intelligence"]}</div>', unsafe_allow_html=True)
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
    c1.plotly_chart(px.histogram(clustered, x="severity", color="customer_segment", title="Severity by segment"), width="stretch", theme=None)
    sig_rows = [(o["label"], k, v) for o in opps if o["actionable"] for k, v in o["signal_profile"].items()]
    if sig_rows:
        c2.plotly_chart(px.density_heatmap(pd.DataFrame(sig_rows, columns=["theme", "signal", "share"]), x="signal", y="theme", z="share",
                                           histfunc="sum", title="Signal profile by theme", color_continuous_scale="Sunsetdark"),
                        width="stretch", theme=None)

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

    next_step()

# ================= PAGE 3 =================
elif page == "Opportunity Economics":
    st.title("Opportunity Economics")
    st.markdown(f'<div class="sf-blurb">{BLURB["Opportunity Economics"]}</div>', unsafe_allow_html=True)
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
    c1.plotly_chart(go.Figure([go.Bar(name="Recoverable value", x=sel.label, y=sel.expected_recoverable_value, marker_color="#7c3aed"),
                               go.Bar(name="Total investment", x=sel.label, y=sel.total_investment, marker_color="#fbbf24")])
                    .update_layout(barmode="group", title="Value vs investment (USD)"), width="stretch", theme=None)
    c2.plotly_chart(px.scatter(ranked, x="effort_weeks", y="net_opportunity_value", size="unique_accounts", color="evidence_sufficiency",
                               hover_name="label", title="Effort vs value (size = accounts)"), width="stretch", theme=None)
    st.subheader("Scenario comparison (bear / base / bull)")
    st.dataframe(scenario_ranks([o for o in opps if o["actionable"] and not o["rejected"]], a), width="stretch")
    st.subheader("Confidence range for a theme")
    pick = st.selectbox("Theme", ranked.cluster_id, format_func=lambda i: by_id[i]["label"], key="econ_pick")
    mc = monte_carlo(by_id[pick], a)
    st.write(f"Model-implied net value: P10 {fmt(mc['p10'])} · P50 {fmt(mc['p50'])} · P90 {fmt(mc['p90'])} · P(net>0) {mc['prob_positive']:.0%} "
             "(conditional on assumed ranges; not an empirical probability).")

    next_step()

# ================= PAGE 4 =================
elif page == "Roadmap Optimizer":
    st.title("Roadmap Optimizer")
    st.markdown(f'<div class="sf-blurb">{BLURB["Roadmap Optimizer"]}</div>', unsafe_allow_html=True)
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

    next_step()

# ================= PAGE 5 =================
elif page == "Scenario Lab":
    st.title("Scenario Lab")
    st.markdown(f'<div class="sf-blurb">{BLURB["Scenario Lab"]}</div>', unsafe_allow_html=True)
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
    st.plotly_chart(go.Figure([go.Bar(y=tn.driver, x=tn.delta_low, orientation="h", name="-30%", marker_color="#ec4899"),
                               go.Bar(y=tn.driver, x=tn.delta_high, orientation="h", name="+30%", marker_color="#14b8a6")])
                    .update_layout(barmode="relative", title="Tornado: change in net value (USD)"), width="stretch", theme=None)
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
                    .add_vline(x=thr_val, line_dash="dash"), width="stretch", theme=None)

    next_step()


# ================= DECISION LOOP (closed-loop vertical slice) =================
elif page == "Decision Loop":
    st.title("Decision Loop")
    signal_panel()
    decs = DB.decisions(WS)
    stages = ["1 · Propose", "2 · Approve", "3 · Ticket + PRD", "4 · Release", "5 · Measure outcome"]
    reached = {"proposed": 1, "approved": 2, "rejected": 1, "in_delivery": 3, "released": 4, "measured": 5, "reverted": 1}
    cur = max([reached[d_.status] for d_ in decs], default=0)
    st.markdown('<div class="sf-steps">' + "".join(
        f'<div class="sf-step {"done" if i < cur else "now" if i == cur else ""}">{"✓ " if i < cur else ""}{s_}</div>'
        for i, s_ in enumerate(stages)) + "</div>", unsafe_allow_html=True)
    st.caption(f"Workspace `{WS}` · stored durably in SQLite ({'file' if DB.durable else 'memory'}). Every action below is "
               "permission-checked, audited and reversible.")

    # ---- 1. propose ----
    with st.container(border=True):
        st.subheader("1 · Propose a decision from evidence")
        pick = st.selectbox("Opportunity", ranked.label, key="loop_pick")
        row = ranked[ranked.label == pick].iloc[0]
        o = by_id[int(row.cluster_id)]
        qd = quality[int(row.cluster_id)]
        cls = {"strong": "sf-good", "moderate": "sf-mid", "weak": "sf-bad"}[qd["grade"]]
        m = st.columns(4)
        m[0].metric("Rank", f"#{int(row.final_rank)}")
        m[1].metric("Accounts", int(row.unique_accounts))
        m[2].metric("Modeled net value", short(row.net_opportunity_value), help="Assumption-based; see Opportunity Economics")
        m[3].metric("Evidence quality", f"{qd['score']:.0%}", help="Weighted share of 8 deterministic checks")
        st.markdown(f'<div class="sf-card {cls}"><b>Evidence: {qd["grade"].upper()}</b> · top-account share {qd["top_account_share"]:.0%} · '
                    f'contradictions {qd["contradiction_share"]:.0%} · sources {qd["sources"]}<br>{qd["next_step"]}</div>', unsafe_allow_html=True)
        if qd["missing"]:
            st.caption("Missing: " + " · ".join(qd["missing"]))
        c1, c2, c3 = st.columns([1, 2, 1])
        owner = c1.text_input("Decision owner", key="loop_owner", placeholder="e.g. Shivani (PM)")
        why = c2.text_input("Rationale (required)", key="loop_why", placeholder="Why this, why now?")
        tgt = c3.number_input("Target adoption %", 1, 100, 50, key="loop_tgt")
        if st.button("Propose decision", key="b_propose", type="primary"):
            try:
                did = LC.propose(DB, WS, row, o, clustered, a, owner, why, tgt / 100, actor=owner or "user")
                st.success(f"Decision #{did} proposed. Next: approve it below.")
                decs = DB.decisions(WS)
            except StoreError as e:
                st.error(str(e))

    if not decs:
        st.info("No decisions yet. Propose one above to unlock approval, ticketing, release and outcome measurement.")
        st.stop()

    # ---- 2. approve ----
    with st.container(border=True):
        st.subheader("2 · Human approval")
        dmap = {d_.id: d_ for d_ in decs}
        sel = st.selectbox("Decision", list(dmap), key="loop_dec", format_func=lambda i: f"#{i} · {dmap[i].cluster_label} · {dmap[i].status}")
        d = DB.get_decision(WS, sel)
        st.markdown(f"**Owner:** {d.owner} · **Status:** `{d.status}` · **Rationale:** {d.rationale}  \n"
                    f"Evidence IDs: {', '.join(d.evidence_ids[:8])}{' …' if len(d.evidence_ids) > 8 else ''}")
        note = st.text_input("Approval / rejection note (required)", key="loop_note")
        b1, b2, b3 = st.columns(3)
        act = None
        if b1.button("Approve decision", key="b_approve", disabled=d.status != "proposed"):
            act = ("approved", note)
        if b2.button("Reject", key="b_reject", disabled=d.status != "proposed"):
            act = ("rejected", note)
        if b3.button("↩ Revert decision", key="b_revert", disabled=d.status in ("proposed", "rejected", "reverted"),
                     help="Reversible: moves it to 'reverted'; history is kept in the audit log"):
            act = ("reverted", note)
        if act:
            try:
                DB.transition(WS, d.id, act[0], d.owner, act[1]); st.rerun()
            except StoreError as e:
                st.error(str(e))

    # ---- 3. ticket + PRD ----
    with st.container(border=True):
        st.subheader("3 · Engineering ticket + PRD (preview before create)")
        od = by_id.get(next((cid for cid, oo in by_id.items() if oo["label"] == d.cluster_label), -1))
        if od is None:
            st.warning("This decision's theme isn't in the current dataset/threshold; switch back to create its ticket.")
        else:
            rowd = ranked[ranked.cluster_id == od["cluster_id"]]
            t_title, t_body = LC.ticket_markdown(d, od)
            with st.expander(f"Preview: {t_title}", expanded=d.status == "approved"):
                st.markdown(t_body)
            existing = DB.tickets(WS, d.id)
            if existing:
                tk = existing[0]
                st.success(f"Ticket #{tk['id']} created {tk['created_at']} UTC.")
                x1, x2, x3 = st.columns(3)
                x1.download_button("Ticket (.md)", f"# {tk['title']}\n\n{tk['body']}", f"ticket_{tk['id']}.md", key="dl_ticket")
                x2.download_button("PRD (.md)", tk["prd"], f"prd_decision_{d.id}.md", key="dl_tprd")
                x3.download_button("PRD (.pdf)", markdown_to_pdf(tk["prd"], "PRD"), f"prd_decision_{d.id}.pdf", "application/pdf", key="dl_tprd_pdf")
            elif st.button("Create ticket", key="b_ticket", disabled=d.status != "approved" or rowd.empty,
                           help="Only approved decisions get tickets"):
                try:
                    LC.create_ticket(DB, WS, d.id, od, rowd.iloc[0], a, d.owner); st.rerun()
                except StoreError as e:
                    st.error(str(e))

    # ---- 4. release ----
    with st.container(border=True):
        st.subheader("4 · Record the release")
        rels = DB.releases(WS, d.id)
        if rels:
            st.success(f"Released {rels[0]['version']} on {rels[0]['released_at']}.")
        else:
            r1, r2, r3 = st.columns([1, 1, 2])
            ver = r1.text_input("Version", "v1.0", key="rel_ver")
            rdate = r2.date_input("Release date", pd.Timestamp("2026-09-01").date(), key="rel_date")
            rnotes = r3.text_input("Release notes", key="rel_notes")
            if st.button("Record release", key="b_release", disabled=d.status != "in_delivery"):
                try:
                    DB.add_release(WS, d.id, ver, str(rdate), rnotes, d.owner); st.rerun()
                except StoreError as e:
                    st.error(str(e))

    # ---- 5. outcome ----
    with st.container(border=True):
        st.subheader("5 · Did it work? Measure the outcome")
        tsrc = st.radio("Telemetry", ["Simulated sample (labelled)", "Upload event CSV"], horizontal=True, key="tel_src")
        events, label = None, ""
        if tsrc.startswith("Simulated"):
            lift = st.slider("Simulation ground truth: true adoption lift injected", -0.2, 0.4, 0.15, 0.01, key="tel_lift",
                             help="Known parameter so you can check the evaluator recovers it. NOT real behaviour.")
            st.markdown('<span class="sf-pill sf-sim">SIMULATED TELEMETRY - illustrative, not measured</span>', unsafe_allow_html=True)
            rd = rels[0]["released_at"] if rels else "2026-09-01"
            events = TM.simulate_events([d.feature_key, "reports", "search", "exports"], d.feature_key, rd, lift)
            label = f"SIMULATED (true lift {lift:+.0%})"
        else:
            st.caption("Columns: " + ", ".join(TM.EVENT_COLUMNS) + f". event_name in {sorted(TM.EVENT_NAMES)}. Feature key for this "
                       f"decision: `{d.feature_key}`.")
            upt = st.file_uploader("Events CSV (max 10 MB)", type=["csv"], key="tel_up")
            if upt is not None:
                try:
                    events = TM.read_events_csv(upt.getvalue()); label = "YOUR TELEMETRY (uploaded)"
                except Exception as e:  # noqa: BLE001
                    st.error(f"That telemetry file isn't cooperating: {e}")
        if events is not None:
            feats = TM.feature_catalog(events)
            feat = st.selectbox("Feature to evaluate", feats, index=feats.index(d.feature_key) if d.feature_key in feats else 0,
                                key="tel_feat") if feats else d.feature_key
            k = st.columns(3)
            k[0].metric("Events", f"{len(events):,}")
            k[1].metric("Accounts", events.account_id.nunique())
            k[2].metric("Activation (7-day)", f"{TM.activation_rate(events):.0%}", help="first key_action within 7 days of signup")
        if st.button("Measure outcome", key="b_measure", type="primary", disabled=events is None or d.status not in ("released", "measured")):
            try:
                ss.last_outcome = LC.measure(DB, WS, d.id, events, label, feat, d.owner)
                if ss.last_outcome["verdict"] != "too_early":
                    st.rerun()
            except (StoreError, ValueError, TM.TelemetryError) as e:
                st.error(str(e))
        outs = DB.outcomes(WS, d.id)
        res = ss.get("last_outcome") if ss.get("last_outcome", {}).get("verdict") == "too_early" else (outs[0] if outs else None)
        if res:
            if res["verdict"] == "too_early":
                st.warning(res["explanation"])
            else:
                vcls = {"worked": "sf-good", "improved_below_target": "sf-mid", "inconclusive": "sf-mid"}.get(res["verdict"], "sf-bad")
                emoji = {"worked": "✓", "improved_below_target": "↗", "inconclusive": "?", "regressed": "↘", "guardrail_breached": "!"}
                st.markdown(f'<div class="sf-card {vcls}"><b>{emoji.get(res["verdict"], "")} Verdict: {res["verdict"].replace("_", " ")}</b> '
                            f'· {res.get("data_label", label)}<br>{res["forecast"]["summary"]}</div>', unsafe_allow_html=True)
                g = pd.DataFrame({"window": ["28 days before", "28 days after"], "adoption": [res["adoption_pre"], res["adoption_post"]]})
                st.plotly_chart(px.bar(g, x="window", y="adoption", color="window", text_auto=".0%",
                                       color_discrete_sequence=["#c4b5fd", "#ec4899"]).update_layout(
                    height=260, showlegend=False, yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=10, b=0)), width="stretch")
                st.caption(f"Method: {res['method']}. Lift 90% CI {res['lift_ci90'][0]:+.1%} to {res['lift_ci90'][1]:+.1%}. "
                           f"Retention {res['retention_prior']:.0%} → {res['retention_post']:.0%}.")

    with st.expander("Workspace audit log (durable)"):
        al = pd.DataFrame(DB.audit_events(WS))
        st.dataframe(al, width="stretch", hide_index=True)
        if not al.empty:
            st.download_button("Export audit log (.csv)", safe_csv(al), "workspace_audit.csv", "text/csv", key="dl_wsaudit")

# ================= PAGE 6 =================
elif page == "Decision Memo & PRD":
    st.title("Executive Decision Memo")
    st.markdown(f'<div class="sf-blurb">{BLURB["Decision Memo & PRD"]}</div>', unsafe_allow_html=True)
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
        c1, c2, c3, _ = st.columns([1, 1, 1, 3])
        fname = name.lower().replace(" ", "_")
        if c1.download_button(f"{name} (.md)", md, f"{fname}.md", key=f"dlmd_{i}"):
            store.log("export", f"{name} .md")
        c2.download_button(f"{name} (.docx)", markdown_to_docx(md), f"{fname}.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=f"dldocx_{i}")
        c3.download_button(f"{name} (.pdf)", markdown_to_pdf(md, name), f"{fname}.pdf", "application/pdf", key=f"dlpdf_{i}")
    full = "\n\n---\n\n".join(docs.values())
    p1, p2, _ = st.columns([1, 1, 2])
    p1.download_button("Download full decision pack (.docx)", markdown_to_docx(full), "signalforge_decision_pack.docx", key="dl_pack")
    p2.download_button("Download full decision pack (.pdf)", markdown_to_pdf(full, "SignalForge decision pack"),
                       "signalforge_decision_pack.pdf", "application/pdf", key="dl_pack_pdf")
    st.success("Decision-ready. Evidence included.")
    st.subheader("Decision audit trail")
    ev = pd.DataFrame(store.events(), columns=["timestamp (UTC)", "event", "detail"])
    st.dataframe(ev, width="stretch", hide_index=True, key="audit_table")
    st.download_button("Export audit trail (.csv)", safe_csv(ev), "audit_trail.csv", "text/csv", key="dl_audit")
    with st.expander("Product metrics for this session (instrumentation, not outcomes)"):
        pm = session_metrics(store.events(), len(ranked), ss.get("t_first"))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Time to first insight", f"{pm['seconds_to_first_insight']:.1f}s" if pm["seconds_to_first_insight"] is not None else "n/a",
                  help="Server-side: session start to first computed analysis.")
        m2.metric("Human corrections", pm["corrections"], help="Renames, merges, rejects, splits, exclusions and rank overrides.")
        m3.metric("Correction rate", f"{pm['correction_rate']:.0%}" if pm["correction_rate"] is not None else "n/a", help="Corrections per theme shown.")
        m4.metric("SIGNAL questions", pm["signal_questions"])
        st.caption("Computed from this session's audit trail only; never stored or shared. Reviewer agreement with the model needs real reviewers and is not claimed.")

st.sidebar.markdown("---")
if spec:
    with st.sidebar.expander("Data provenance"):
        st.markdown(f"**{spec.name}**  \nLicense: {spec.license}  \nSource: {spec.url}  \n{spec.provenance}  \n*Limits:* {spec.limitations}")
