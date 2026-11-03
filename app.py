from __future__ import annotations

from pathlib import Path
import io
import pandas as pd
import plotly.express as px
import streamlit as st

from src.preprocessing import (
    DataValidationError, make_basket_matrix, preprocess_transactions, validate_columns
)
from src.mining import run_apriori, exportable_itemsets, exportable_rules, format_itemset
from src.recommendations import recommend_products

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "sample_transactions.csv"
st.set_page_config(
    page_title="Smart Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light, high-contrast retail analytics theme.
st.markdown("""
<style>
:root { color-scheme: light; }
.block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1500px; }
[data-testid="stAppViewContainer"] { background: #f5f7fb; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5eaf2; }
h1, h2, h3 { color: #14213d; letter-spacing: -0.02em; }
p, label, .stMarkdown { color: #344054; }
.hero { background: linear-gradient(120deg,#102a43,#176b87); padding: 1.35rem 1.5rem; border-radius: 18px; color: white; margin-bottom: 1rem; }
.hero h1 { color: white; margin: 0; font-size: 2rem; }
.hero p { color: #e6f4f7; margin: .4rem 0 0 0; }
.kpi { background: white; border: 1px solid #e5eaf2; border-radius: 14px; padding: 1rem 1.1rem; min-height: 110px; box-shadow: 0 3px 12px rgba(16,42,67,.035); }
.kpi-label { font-size: .82rem; color: #667085; margin-bottom: .45rem; }
.kpi-value { font-size: 1.8rem; font-weight: 750; color: #102a43; line-height: 1.1; }
.kpi-note { font-size: .78rem; color: #667085; margin-top: .45rem; }
.section-note { color: #667085; font-size: .92rem; }
div[data-testid="stDataFrame"] { border: 1px solid #e5eaf2; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


def init_state():
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = pd.read_csv(SAMPLE_PATH)
        st.session_state.source_name = "Built-in demo dataset"
    if "analysis" not in st.session_state:
        st.session_state.analysis = None


def fmt(value, digits=2):
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def kpi(label, value, note="Calculated from current data"):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True
    )


def analyze(raw_df, support, max_len, confidence, lift):
    cleaned, stats = preprocess_transactions(raw_df)
    basket = make_basket_matrix(cleaned)
    itemsets, rules = run_apriori(
        basket, min_support=support, max_len=max_len,
        min_confidence=confidence, min_lift=lift
    )
    product_counts = (
        cleaned.groupby("Description")["InvoiceNo"].nunique()
        .sort_values(ascending=False).rename("Transactions").reset_index()
    )
    return {
        "cleaned": cleaned, "stats": stats, "basket": basket,
        "itemsets": itemsets, "rules": rules, "product_counts": product_counts,
        "support": support, "max_len": max_len, "confidence": confidence, "lift": lift,
    }


def render_overview(a):
    st.markdown('<div class="hero"><h1>Smart Market Basket Analysis</h1>'
                '<p>Discover what shoppers buy together and turn transaction patterns into practical product recommendations.</p></div>',
                unsafe_allow_html=True)
    st.caption(f"Active data source: **{st.session_state.source_name}**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Transactions", f"{a['basket'].shape[0]:,}", "Unique invoice numbers")
    with c2: kpi("Unique products", f"{a['basket'].shape[1]:,}", "After cleaning")
    with c3: kpi("Frequent itemsets", f"{len(a['itemsets']):,}", f"Support ≥ {a['support']:.0%}")
    with c4: kpi("Association rules", f"{len(a['rules']):,}", f"Confidence ≥ {a['confidence']:.0%}")
    strongest = fmt(a["rules"].iloc[0]["lift"], 2) if not a["rules"].empty else "—"
    with c5: kpi("Strongest lift", strongest, "Highest lift among retained rules")

    st.divider()
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Most frequently purchased products")
        counts = a["product_counts"].head(12).sort_values("Transactions")
        if counts.empty:
            st.info("No products remain after cleaning. Upload a valid transaction file.")
        else:
            fig = px.bar(counts, x="Transactions", y="Description", orientation="h",
                         color="Transactions", color_continuous_scale="Teal",
                         labels={"Description": "Product", "Transactions": "Invoices"})
            fig.update_layout(height=390, margin=dict(l=0,r=10,t=10,b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Association rules: confidence vs lift")
        if a["rules"].empty:
            st.info("No rules meet the current thresholds. Lower minimum support/confidence or minimum lift.")
        else:
            plot_df = a["rules"].head(500).copy()
            fig = px.scatter(plot_df, x="confidence", y="lift", size="support",
                             hover_data=["antecedents_text", "consequents_text"],
                             labels={"confidence": "Confidence", "lift": "Lift", "support": "Support"},
                             color="lift", color_continuous_scale="Viridis")
            fig.update_layout(height=390, margin=dict(l=0,r=10,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strongest association rules")
    if a["rules"].empty:
        st.info("No association rules are available for the selected mining settings.")
    else:
        view = a["rules"][["antecedents_text","consequents_text","support","confidence","lift"]].head(10).copy()
        view.columns = ["If basket contains", "Then associated with", "Support", "Confidence", "Lift"]
        st.dataframe(view, use_container_width=True, hide_index=True,
                     column_config={
                         "Support": st.column_config.NumberColumn(format="%.3f"),
                         "Confidence": st.column_config.NumberColumn(format="%.3f"),
                         "Lift": st.column_config.NumberColumn(format="%.3f"),
                     })


def render_upload():
    st.title("📂 Data Upload")
    st.write("Upload a CSV with at least `InvoiceNo`, `Description`, and `Quantity`. The built-in sample remains available at any time.")
    uploaded = st.file_uploader("Choose transaction CSV", type=["csv"], help="UTF-8 CSV recommended.")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Load built-in demo dataset", use_container_width=True):
            st.session_state.raw_df = pd.read_csv(SAMPLE_PATH)
            st.session_state.source_name = "Built-in demo dataset"
            st.session_state.analysis = None
            st.success("Demo dataset loaded.")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            raw.columns = [str(c).strip() for c in raw.columns]
            validate_columns(raw)
            st.session_state.raw_df = raw
            st.session_state.source_name = uploaded.name
            st.session_state.analysis = None
            st.success(f"Loaded {len(raw):,} rows from {uploaded.name}.")
        except (pd.errors.EmptyDataError, UnicodeDecodeError, pd.errors.ParserError, DataValidationError) as exc:
            st.error(f"Could not read this CSV: {exc}")
    st.subheader("Raw data preview")
    raw_df = st.session_state.raw_df
    st.caption(f"{len(raw_df):,} rows · {len(raw_df.columns)} columns")
    st.dataframe(raw_df.head(100), use_container_width=True, hide_index=True)
    missing = [c for c in ["InvoiceNo","Description","Quantity"] if c not in raw_df.columns]
    if missing:
        st.error("Required columns missing: " + ", ".join(missing))


def render_explorer(a):
    st.title("🔎 Data Explorer")
    st.write("Inspect the uploaded rows, cleaned transactions, cleaning decisions, and a small slice of the basket matrix.")
    raw = st.session_state.raw_df
    tab1, tab2, tab3, tab4 = st.tabs(["Raw preview", "Cleaned preview", "Cleaning report", "Basket matrix"])
    with tab1:
        st.dataframe(raw.head(100), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(a["cleaned"].head(100), use_container_width=True, hide_index=True)
        st.download_button("Download cleaned transactions CSV",
                           a["cleaned"].to_csv(index=False).encode("utf-8"),
                           "cleaned_transactions.csv", "text/csv")
    with tab3:
        stats_df = pd.DataFrame([a["stats"].to_dict()]).T.reset_index()
        stats_df.columns = ["Cleaning metric", "Rows"]
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        st.caption("Duplicate policy: for each invoice-product pair, keep the first valid row. Quantities are not summed.")
    with tab4:
        st.caption(f"Full matrix shape: {a['basket'].shape[0]:,} invoices × {a['basket'].shape[1]:,} products. Preview limited to 30 × 30.")
        st.dataframe(a["basket"].iloc[:30, :30], use_container_width=True)
        st.download_button("Download full basket matrix CSV",
                           a["basket"].reset_index().to_csv(index=False).encode("utf-8"),
                           "basket_matrix.csv", "text/csv")


def render_itemsets(a):
    st.title("🧩 Frequent Itemsets")
    st.write("An itemset is a group of products that appears together in a sufficient proportion of invoices.")
    if a["itemsets"].empty:
        st.warning("No frequent itemsets found. Try reducing minimum support or use a larger dataset.")
        return
    view = a["itemsets"].copy()
    view["Products"] = view["itemsets"].apply(format_itemset)
    view = view[["Products", "length", "support"]]
    view.columns = ["Products", "Item count", "Support"]
    min_sup, max_sup = float(view["Support"].min()), float(view["Support"].max())
    chosen = st.slider("Filter by support", min_value=0.0, max_value=1.0,
                       value=(min_sup, max_sup), step=0.01)
    view = view[view["Support"].between(chosen[0], chosen[1])]
    st.dataframe(view, use_container_width=True, hide_index=True,
                 column_config={"Support": st.column_config.NumberColumn(format="%.4f")})
    st.download_button("Download frequent itemsets CSV",
                       exportable_itemsets(a["itemsets"]).to_csv(index=False).encode("utf-8"),
                       "frequent_itemsets.csv", "text/csv")


def render_rules(a):
    st.title("🔗 Association Rules")
    st.write("Rules have the form **antecedent → consequent**. Confidence estimates how often the consequent appears when the antecedent appears; lift compares this with the consequent's baseline frequency.")
    rules = a["rules"]
    if rules.empty:
        st.warning("No rules meet your current thresholds. Adjust mining settings in the sidebar.")
        return
    min_conf = st.slider("Minimum confidence for display", 0.0, 1.0, float(a["confidence"]), 0.01)
    min_lift = st.slider("Minimum lift for display", 0.0, max(5.0, float(rules["lift"].max())),
                         float(min(float(a["lift"]), max(5.0, float(rules["lift"].max())))), 0.1)
    view = rules.loc[(rules["confidence"] >= min_conf) & (rules["lift"] >= min_lift)].copy()
    view = view[["antecedents_text","consequents_text","support","confidence","lift"]]
    view.columns = ["Antecedent", "Consequent", "Support", "Confidence", "Lift"]
    st.caption(f"{len(view):,} rules match these display filters.")
    st.dataframe(view, use_container_width=True, hide_index=True,
                 column_config={
                     "Support": st.column_config.NumberColumn(format="%.4f"),
                     "Confidence": st.column_config.NumberColumn(format="%.4f"),
                     "Lift": st.column_config.NumberColumn(format="%.4f"),
                 })
    st.download_button("Download association rules CSV",
                       exportable_rules(rules).to_csv(index=False).encode("utf-8"),
                       "association_rules.csv", "text/csv")


def render_recommendations(a):
    st.title("✨ Product Recommendations")
    st.write("Select the products already in a shopper's basket. The system finds rules whose complete antecedent is present, removes products already selected, and ranks remaining products by confidence, then lift.")
    products = sorted(a["basket"].columns.tolist())
    if not products:
        st.warning("No products available. Load a valid dataset first.")
        return
    selected = st.multiselect("Products already in basket", products, placeholder="Choose one or more products")
    col1, col2, col3 = st.columns(3)
    with col1:
        top_n = st.slider("Maximum recommendations", 1, 20, 10)
    with col2:
        min_conf = st.slider("Recommendation confidence", 0.0, 1.0, float(a["confidence"]), 0.05)
    with col3:
        min_lift = st.slider("Recommendation lift", 0.0, 10.0, float(a["lift"]), 0.1)
    if not selected:
        st.info("Choose at least one product to see rule-based recommendations.")
    elif a["rules"].empty:
        st.warning("There are no association rules. Lower the mining thresholds in the sidebar.")
    else:
        recs = recommend_products(selected, a["rules"], top_n, min_conf, min_lift)
        if recs.empty:
            st.info("No recommendations matched this basket and these thresholds. Try selecting another combination or lowering recommendation filters.")
        else:
            st.success(f"{len(recs)} recommendation(s) found.")
            st.dataframe(recs, use_container_width=True, hide_index=True,
                         column_config={
                             "Confidence": st.column_config.NumberColumn(format="%.3f"),
                             "Lift": st.column_config.NumberColumn(format="%.3f"),
                             "Rule Support": st.column_config.NumberColumn(format="%.3f"),
                         })
            st.download_button("Download recommendations CSV",
                               recs.to_csv(index=False).encode("utf-8"),
                               "product_recommendations.csv", "text/csv")


def main():
    init_state()
    with st.sidebar:
        st.markdown("## 🛒 Smart Market Basket")
        st.caption("Retail analytics · Data mining")
        page = st.radio("Navigation", [
            "Overview", "Data Upload", "Data Explorer", "Frequent Itemsets",
            "Association Rules", "Product Recommendations"
        ], label_visibility="collapsed")
        st.divider()
        st.subheader("Mining settings")
        support = st.slider("Minimum support", 0.01, 0.80, 0.10, 0.01,
                            help="Fraction of invoices containing an itemset.")
        max_len = st.slider("Maximum itemset length", 1, 5, 3, 1)
        confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.30, 0.05)
        lift = st.slider("Minimum lift", 0.0, 5.0, 1.0, 0.1)
        run = st.button("Run analysis", type="primary", use_container_width=True)
        st.caption("For larger files, begin with higher support and max itemset length 2–3.")

    if run or st.session_state.analysis is None:
        try:
            with st.spinner("Cleaning transactions and mining association patterns…"):
                st.session_state.analysis = analyze(
                    st.session_state.raw_df, support, max_len, confidence, lift
                )
        except (DataValidationError, MemoryError, ValueError) as exc:
            st.session_state.analysis = None
            st.error(str(exc))
            st.info("Check the required columns and use a manageable dataset size.")
            if page != "Data Upload":
                return
    a = st.session_state.analysis
    if a is None:
        return

    if page == "Overview":
        render_overview(a)
    elif page == "Data Upload":
        render_upload()
    elif page == "Data Explorer":
        render_explorer(a)
    elif page == "Frequent Itemsets":
        render_itemsets(a)
    elif page == "Association Rules":
        render_rules(a)
    elif page == "Product Recommendations":
        render_recommendations(a)

    with st.sidebar:
        st.divider()
        st.caption("Built with Streamlit · Pandas · mlxtend · Plotly")


if __name__ == "__main__":
    main()
