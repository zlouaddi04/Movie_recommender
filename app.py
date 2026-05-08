"""
🎬 Movie Recommendation System — Streamlit Frontend
────────────────────────────────────────────────────
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd

# ── Backend imports ────────────────────────────────────────────────────────────
from src.data import load_and_preprocess
from src.models import build_model
from src.recommender import recommend

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES  — Netflix-inspired cinematic dark theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ── Google Fonts ────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&display=swap');

    /* ── Design tokens ───────────────────────────────────────────────── */
    :root {
        --bg:          #0f0f0f;
        --surface:     #181818;
        --surface-2:   #222222;
        --border:      #2c2c2c;
        --red:         #e50914;
        --red-hover:   #f40612;
        --gold:        #f5c518;
        --text:        #ffffff;
        --text-2:      #b3b3b3;
        --text-3:      #6b6b6b;
        --card-r:      6px;
        --panel-r:     10px;
        --transition:  0.25s cubic-bezier(0.4,0,0.2,1);
    }

    /* ── Reset Streamlit chrome ──────────────────────────────────────── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Outfit', sans-serif !important;
    }
    [data-testid="stHeader"]  { background: transparent !important; display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    footer                    { visibility: hidden !important; }
    .block-container          { padding: 0 !important; max-width: 100% !important; }

    /* ── Scrollbar ───────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #444; }

    /* ════════════════════════════════════════════════════════════════════
       NAVBAR
    ════════════════════════════════════════════════════════════════════ */
    .nav-bar {
        position: sticky;
        top: 0;
        z-index: 999;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 4%;
        height: 68px;
        background: linear-gradient(180deg, rgba(0,0,0,0.9) 0%, rgba(0,0,0,0) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .nav-logo {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2rem;
        letter-spacing: 0.12em;
        color: var(--red) !important;
        text-decoration: none;
        -webkit-text-fill-color: var(--red);
    }
    .nav-links {
        display: flex;
        gap: 2rem;
        list-style: none;
        margin: 0; padding: 0;
    }
    .nav-links li {
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--text-2);
        cursor: pointer;
        transition: color var(--transition);
    }
    .nav-links li:hover { color: var(--text); }
    .nav-links li.active { color: var(--text); font-weight: 600; }

    /* ════════════════════════════════════════════════════════════════════
       HERO BANNER
    ════════════════════════════════════════════════════════════════════ */
    .hero {
        position: relative;
        width: 100%;
        min-height: 420px;
        padding: 5% 4% 4%;
        background:
            linear-gradient(77deg, rgba(0,0,0,.95) 30%, transparent 70%),
            linear-gradient(180deg, rgba(0,0,0,.4) 0%, var(--bg) 100%),
            url('https://image.tmdb.org/t/p/original/cinematic_banner_fallback.jpg') center/cover no-repeat;
        background-color: #1a0a0a;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }
    .hero::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse 80% 60% at 70% 50%, rgba(229,9,20,.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-eyebrow {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--red);
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: clamp(3rem, 7vw, 5.5rem);
        letter-spacing: 0.04em;
        line-height: 0.95;
        color: var(--text);
        margin-bottom: 1rem;
        text-shadow: 0 4px 32px rgba(0,0,0,.8);
    }
    .hero-desc {
        font-size: clamp(0.85rem, 1.5vw, 1rem);
        font-weight: 300;
        color: var(--text-2);
        max-width: 480px;
        line-height: 1.65;
        margin-bottom: 1.8rem;
    }
    .hero-badges {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-bottom: 2rem;
    }
    .badge {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.28rem 0.75rem;
        border-radius: 4px;
        border: 1px solid rgba(255,255,255,.2);
        color: var(--text-2);
    }
    .badge-gold {
        border-color: var(--gold);
        color: var(--gold);
    }

    /* ════════════════════════════════════════════════════════════════════
       MAIN CONTENT WRAPPER
    ════════════════════════════════════════════════════════════════════ */
    .content-wrap {
        padding: 2.5rem 4%;
    }

    /* ════════════════════════════════════════════════════════════════════
       STATS ROW
    ════════════════════════════════════════════════════════════════════ */
    .stats-row {
        display: flex;
        gap: 1px;
        background: var(--border);
        border-radius: var(--panel-r);
        overflow: hidden;
        margin-bottom: 2.5rem;
    }
    .stat-cell {
        flex: 1;
        background: var(--surface);
        padding: 1.1rem 1.4rem;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 0.2rem;
        transition: background var(--transition);
    }
    .stat-cell:hover { background: var(--surface-2); }
    .stat-num {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.1rem;
        letter-spacing: 0.05em;
        color: var(--red);
        line-height: 1;
    }
    .stat-tag {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-3);
    }

    /* ════════════════════════════════════════════════════════════════════
       SECTION HEADER
    ════════════════════════════════════════════════════════════════════ */
    .section-head {
        display: flex;
        align-items: baseline;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.01em;
    }
    .section-more {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-3);
        cursor: pointer;
        letter-spacing: 0.04em;
        transition: color var(--transition);
    }
    .section-more:hover { color: var(--text); }

    /* ════════════════════════════════════════════════════════════════════
       CONTROL PANEL
    ════════════════════════════════════════════════════════════════════ */
    .control-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--panel-r);
        padding: 1.6rem 2rem;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
    }
    .control-panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--red), transparent 60%);
    }
    .panel-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--text-3);
        margin-bottom: 0.5rem;
        display: block;
    }

    /* ── Streamlit widget overrides ──────────────────────────────────── */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"]  > div {
        background-color: var(--surface-2) !important;
        border-color: var(--border) !important;
        border-radius: 6px !important;
        color: var(--text) !important;
        font-family: 'Outfit', sans-serif !important;
    }
    div[data-baseweb="select"] * { color: var(--text) !important; font-family: 'Outfit', sans-serif !important; }
    div[data-baseweb="popover"] { background: var(--surface-2) !important; border-color: var(--border) !important; }
    li[role="option"] { background: var(--surface-2) !important; }
    li[role="option"]:hover { background: var(--surface) !important; }

    .stSlider > div > div { color: var(--text-2) !important; }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--red) !important;
        border-color: var(--red) !important;
        width: 18px !important; height: 18px !important;
    }
    .stSlider [data-baseweb="slider"] [data-testid="stSliderTrackFill"] {
        background: var(--red) !important;
    }
    .stSlider [data-baseweb="slider"] div[aria-hidden="true"] {
        background: var(--border) !important;
    }

    label,
    .stSlider label,
    .stSelectbox label {
        color: var(--text-3) !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.14em !important;
        font-family: 'Outfit', sans-serif !important;
    }

    /* ── Recommend button ─────────────────────────────────────────────── */
    div[data-testid="stButton"] > button {
        background: var(--red) !important;
        color: #fff !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.7rem 2.2rem !important;
        transition: background var(--transition), transform var(--transition), box-shadow var(--transition) !important;
        box-shadow: 0 4px 20px rgba(229,9,20,.3) !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: var(--red-hover) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 28px rgba(229,9,20,.45) !important;
    }
    div[data-testid="stButton"] > button:active {
        transform: translateY(0) !important;
    }

    /* ════════════════════════════════════════════════════════════════════
       MOVIE CARDS
    ════════════════════════════════════════════════════════════════════ */
    .results-grid {
        display: flex;
        flex-direction: column;
        gap: 0;
    }

    .movie-card {
        display: flex;
        align-items: center;
        gap: 1.2rem;
        padding: 0.9rem 1.2rem;
        background: transparent;
        border-radius: var(--card-r);
        border: 1px solid transparent;
        transition: background var(--transition), border-color var(--transition), transform var(--transition);
        cursor: default;
        position: relative;
        overflow: hidden;
    }
    .movie-card:hover {
        background: var(--surface);
        border-color: var(--border);
        transform: translateX(4px);
    }
    .movie-card::after {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: var(--red);
        border-radius: 0 2px 2px 0;
        transform: scaleY(0);
        transition: transform var(--transition);
        transform-origin: center;
    }
    .movie-card:hover::after {
        transform: scaleY(1);
    }

    /* Rank number */
    .card-rank {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2rem;
        color: rgba(255,255,255,.08);
        min-width: 2.8rem;
        text-align: center;
        line-height: 1;
        user-select: none;
        transition: color var(--transition);
    }
    .movie-card:hover .card-rank {
        color: rgba(229,9,20,.35);
    }

    /* Poster */
    .card-poster {
        width: 52px;
        height: 78px;
        border-radius: 5px;
        object-fit: cover;
        flex-shrink: 0;
        background: var(--border);
        box-shadow: 0 4px 12px rgba(0,0,0,.5);
        transition: box-shadow var(--transition), transform var(--transition);
    }
    .movie-card:hover .card-poster {
        box-shadow: 0 6px 20px rgba(0,0,0,.7);
        transform: scale(1.04);
    }
    .card-poster-placeholder {
        width: 52px;
        height: 78px;
        border-radius: 5px;
        flex-shrink: 0;
        background: var(--surface-2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        border: 1px solid var(--border);
    }

    /* Info block */
    .card-info { flex: 1; min-width: 0; }
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 0.28rem;
        letter-spacing: -0.01em;
    }
    .card-meta {
        font-size: 0.78rem;
        color: var(--text-3);
        font-weight: 400;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
    }
    .meta-sep {
        color: var(--border);
        font-size: 0.6rem;
    }
    .meta-rating {
        color: var(--gold);
        font-weight: 600;
    }

    /* Score pill + bar */
    .score-wrap {
        flex-shrink: 0;
        text-align: right;
    }
    .score-pill {
        display: inline-block;
        background: rgba(229,9,20,.12);
        border: 1px solid rgba(229,9,20,.3);
        color: #ff6b6b;
        font-weight: 700;
        font-size: 0.8rem;
        border-radius: 5px;
        padding: 0.22rem 0.65rem;
        letter-spacing: 0.04em;
        font-family: 'Outfit', monospace;
        transition: background var(--transition), border-color var(--transition);
    }
    .movie-card:hover .score-pill {
        background: rgba(229,9,20,.2);
        border-color: rgba(229,9,20,.5);
    }
    .score-bar-track {
        height: 2px;
        background: var(--border);
        border-radius: 1px;
        margin-top: 0.5rem;
        width: 80px;
        margin-left: auto;
    }
    .score-bar-fill {
        height: 2px;
        border-radius: 1px;
        background: var(--red);
        transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
    }

    /* Divider between cards (subtle) */
    .card-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border) 20%, var(--border) 80%, transparent);
        margin: 0 1.2rem;
        opacity: 0.5;
    }

    /* ════════════════════════════════════════════════════════════════════
       EMPTY STATE
    ════════════════════════════════════════════════════════════════════ */
    .empty-state {
        margin-top: 1.5rem;
        padding: 4rem 2rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--panel-r);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .empty-state::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse 60% 50% at 50% 80%, rgba(229,9,20,.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .empty-icon {
        font-size: 3.5rem;
        display: block;
        margin-bottom: 1rem;
        filter: grayscale(0.3);
    }
    .empty-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.5rem;
        letter-spacing: -0.01em;
    }
    .empty-sub {
        font-size: 0.85rem;
        color: var(--text-3);
        font-weight: 300;
        line-height: 1.6;
        max-width: 380px;
        margin: 0 auto;
    }
    .empty-formula {
        display: inline-block;
        margin-top: 1.4rem;
        padding: 0.4rem 1rem;
        background: rgba(229,9,20,.08);
        border: 1px solid rgba(229,9,20,.18);
        border-radius: 4px;
        font-size: 0.75rem;
        color: rgba(229,9,20,.7);
        letter-spacing: 0.06em;
        font-family: 'Outfit', monospace;
    }

    /* ════════════════════════════════════════════════════════════════════
       SPINNER / ALERTS
    ════════════════════════════════════════════════════════════════════ */
    .stSpinner > div { border-top-color: var(--red) !important; }
    [data-testid="stSpinner"] p { color: var(--text-2) !important; font-family: 'Outfit', sans-serif !important; }
    .stAlert {
        border-radius: var(--panel-r) !important;
        background: var(--surface) !important;
        border-color: var(--border) !important;
        color: var(--text-2) !important;
    }

    /* ── Results header ───────────────────────────────────────────────── */
    .results-meta {
        font-size: 0.8rem;
        color: var(--text-3);
        font-weight: 400;
        margin-bottom: 0.6rem;
        letter-spacing: 0.02em;
    }
    .results-meta strong {
        color: var(--text-2);
        font-weight: 600;
    }

    /* ── Responsive tweaks ───────────────────────────────────────────── */
    @media (max-width: 640px) {
        .nav-links       { display: none; }
        .hero            { min-height: 280px; padding: 6% 5% 5%; }
        .hero-title      { font-size: 2.5rem; }
        .content-wrap    { padding: 1.5rem 5%; }
        .stats-row       { flex-direction: column; gap: 1px; }
        .card-rank       { display: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED BACKEND CALLS  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_and_preprocess()


@st.cache_resource(show_spinner=False)
def get_model(df: pd.DataFrame):
    return build_model(df)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS  (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════

def _score_bar(score: float, max_score: float = 1.0) -> str:
    pct = min(score / max_score, 1.0) * 100
    return (
        f'<div class="score-bar-track">'
        f'  <div class="score-bar-fill" style="width:{pct:.1f}%"></div>'
        f'</div>'
    )


def _poster_html(row: pd.Series) -> str:
    for col in ("poster_url", "poster_path", "backdrop_path"):
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            url = str(row[col]).strip()
            if url.startswith("/"):
                url = f"https://image.tmdb.org/t/p/w92{url}"
            return f'<img class="card-poster" src="{url}" alt="poster" />'
    return '<div class="card-poster-placeholder">🎞️</div>'


def _extra_meta(row: pd.Series) -> str:
    parts = []
    for col, label, fmt in [
        ("release_date", "", "year"),
        ("vote_average", "⭐ ", "rating"),
        ("genres", "", "genres"),
    ]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            val = str(row[col]).strip()
            if fmt == "year":
                val = val[:4]
            elif fmt == "rating":
                try:
                    val = f'<span class="meta-rating">{label}{float(val):.1f}</span>'
                except ValueError:
                    pass
            elif fmt == "genres" and len(val) > 36:
                val = val[:36] + "…"
            else:
                val = f"{label}{val}"
            parts.append(val)

    if not parts:
        return '<span style="color:var(--text-3)">No metadata</span>'

    result = ""
    for i, p in enumerate(parts):
        if i > 0:
            result += ' <span class="meta-sep">◆</span> '
        result += p
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:

    # ── Navbar ─────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="nav-bar">
            <span class="nav-logo">CINEMATCH</span>
            <ul class="nav-links">
                <li class="active">Discover</li>
                <li>Top Picks</li>
                <li>Genres</li>
                <li>About</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data + model ──────────────────────────────────────────────────
    with st.spinner("Loading movie database…"):
        df = get_data()

    with st.spinner("Warming up the recommendation engine…"):
        model = get_model(df)

    title_col = next(
        (c for c in ("title", "original_title", "movie_title", "name") if c in df.columns),
        df.columns[0],
    )
    all_titles: list[str] = sorted(df[title_col].dropna().unique().tolist())

    # ── Hero banner ────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-eyebrow">✦ AI-Powered Recommendations</div>
            <div class="hero-title">Find Your<br>Next Obsession</div>
            <div class="hero-desc">
                Content-based matching powered by TF-IDF + KNN. Pick a film,
                get instant personalized picks ranked by hybrid similarity score.
            </div>
            <div class="hero-badges">
                <span class="badge">TF-IDF + KNN</span>
                <span class="badge">Hybrid Scoring</span>
                <span class="badge badge-gold">⭐ Rating Weighted</span>
                <span class="badge">{len(df):,} Movies</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Content wrapper start ──────────────────────────────────────────────
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

    # ── Stats row ──────────────────────────────────────────────────────────
    genres_col = next((c for c in ("genres",) if c in df.columns), None)
    genre_count = (
        df[genres_col].dropna().apply(lambda x: len(str(x).split(","))).sum()
        if genres_col else 0
    )

    st.markdown(
        f"""
        <div class="stats-row">
            <div class="stat-cell">
                <span class="stat-num">{len(df):,}</span>
                <span class="stat-tag">Movies Indexed</span>
            </div>
            <div class="stat-cell">
                <span class="stat-num">{"—" if not genre_count else f"{int(genre_count):,}"}</span>
                <span class="stat-tag">Genre Tags</span>
            </div>
            <div class="stat-cell">
                <span class="stat-num">TF·KNN</span>
                <span class="stat-tag">Algorithm</span>
            </div>
            <div class="stat-cell">
                <span class="stat-num">0.7/0.2/0.1</span>
                <span class="stat-tag">Score Weights</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Control panel ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-head">'
        '  <span class="section-title">Search & Discover</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)

        left, right = st.columns([3, 1], gap="large")

        with left:
            selected_movie: str = st.selectbox(
                "Choose a movie",
                options=all_titles,
                index=0,
                placeholder="Start typing a title…",
            )

        with right:
            top_n: int = st.slider(
                "Recommendations",
                min_value=3,
                max_value=20,
                value=6,
                step=1,
            )

        recommend_btn = st.button("▶  Get Recommendations", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Results ────────────────────────────────────────────────────────────
    if recommend_btn:

        if not selected_movie:
            st.warning("Please select a movie first.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        try:
            with st.spinner(f"Scanning {len(df):,} films for matches…"):
                matrix, knn_model = get_model(df)

                results = recommend(
                    movie_title=selected_movie,
                    movies=df,
                    model=knn_model,
                    matrix=matrix,
                    top_n=top_n,
                )

                if isinstance(results, list):
                    results = pd.DataFrame(results, columns=["title", "final_score"])

        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if results is None or len(results) == 0:
            st.info("No recommendations found. Try a different movie.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        results = results.copy()

        if "final_score" not in results.columns:
            score_col = next((c for c in results.columns if "score" in c.lower()), None)
            if score_col:
                results = results.rename(columns={score_col: "final_score"})
            else:
                results["final_score"] = 0.0

        results = results.sort_values("final_score", ascending=False).reset_index(drop=True)
        results = results.merge(df, on="title", how="left")
        max_score = results["final_score"].max() or 1.0

        st.markdown(
            f'<div class="section-head" style="margin-top:0.5rem;">'
            f'  <span class="section-title">Similar to <em style="color:var(--red)">{selected_movie}</em></span>'
            f'</div>'
            f'<p class="results-meta">Showing <strong>{len(results)}</strong> results · ranked by hybrid similarity score</p>',
            unsafe_allow_html=True,
        )

        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            movie_name  = row.get("title", "Unknown")
            score_val   = float(row.get("final_score", 0.0))
            poster_html = _poster_html(row)
            meta_html   = _extra_meta(row)
            bar_html    = _score_bar(score_val, max_score)

            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="card-rank">{rank}</div>
                    {poster_html}
                    <div class="card-info">
                        <div class="card-title">{movie_name}</div>
                        <div class="card-meta">{meta_html}</div>
                    </div>
                    <div class="score-wrap">
                        <span class="score-pill">{score_val:.3f}</span>
                        {bar_html}
                    </div>
                </div>
                {"" if rank == len(results) else '<div class="card-divider"></div>'}
                """,
                unsafe_allow_html=True,
            )

    else:
        st.markdown(
            """
            <div class="empty-state">
                <span class="empty-icon">🍿</span>
                <div class="empty-title">Your next favourite film is one click away</div>
                <div class="empty-sub">
                    Select any movie from the dropdown above and hit
                    <strong style="color:var(--red)">Get Recommendations</strong>
                    to discover films you'll love.
                </div>
                <div class="empty-formula">score = 0.7 × similarity + 0.2 × rating + 0.1 × success</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)  # .content-wrap


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()