"""Apriori frequent itemsets and association-rule generation."""
from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from .preprocessing import DataValidationError


def run_apriori(
    basket: pd.DataFrame,
    min_support: float = 0.05,
    max_len: int = 3,
    min_confidence: float = 0.30,
    min_lift: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mine frequent itemsets and rules; empty results are valid outcomes."""
    if basket is None or basket.empty:
        return pd.DataFrame(columns=["support", "itemsets"]), empty_rules()
    if not 0 < float(min_support) <= 1:
        raise DataValidationError("Minimum support must be greater than 0 and at most 1.")
    if not 1 <= int(max_len) <= 8:
        raise DataValidationError("Maximum itemset length must be between 1 and 8.")
    if not 0 <= float(min_confidence) <= 1:
        raise DataValidationError("Minimum confidence must be between 0 and 1.")
    if float(min_lift) < 0:
        raise DataValidationError("Minimum lift cannot be negative.")

    # mlxtend requires bool or numeric data; keep columns and index unchanged.
    bool_basket = basket.astype(bool)
    try:
        itemsets = apriori(
            bool_basket, min_support=float(min_support),
            use_colnames=True, max_len=int(max_len), low_memory=True
        )
    except (ValueError, MemoryError) as exc:
        raise DataValidationError(f"Apriori could not process this basket: {exc}") from exc

    if itemsets.empty:
        itemsets = pd.DataFrame(columns=["support", "itemsets", "length"])
        return itemsets, empty_rules()

    itemsets["length"] = itemsets["itemsets"].apply(len)
    itemsets = itemsets.sort_values(
        ["support", "length"], ascending=[False, True]
    ).reset_index(drop=True)

    if len(itemsets) < 2:
        return itemsets, empty_rules()

    try:
        rules = association_rules(
            itemsets, metric="confidence", min_threshold=float(min_confidence),
            num_itemsets=len(bool_basket)
        )
    except TypeError:
        # Compatibility with mlxtend versions that do not accept num_itemsets.
        rules = association_rules(
            itemsets, metric="confidence", min_threshold=float(min_confidence)
        )
    if rules.empty:
        return itemsets, empty_rules()

    rules = rules.loc[rules["lift"] >= float(min_lift)].copy()
    rules["antecedents_text"] = rules["antecedents"].apply(format_itemset)
    rules["consequents_text"] = rules["consequents"].apply(format_itemset)
    rules = rules.sort_values(
        ["lift", "confidence", "support"], ascending=False
    ).reset_index(drop=True)
    return itemsets, rules


def empty_rules() -> pd.DataFrame:
    """Return an empty rules table with the expected columns."""
    return pd.DataFrame(columns=[
        "antecedents", "consequents", "antecedent support",
        "consequent support", "support", "confidence", "lift",
        "leverage", "conviction", "antecedents_text", "consequents_text"
    ])


def format_itemset(items) -> str:
    """Format a frozenset/set for display and CSV export."""
    return ", ".join(sorted(str(item) for item in items))


def exportable_itemsets(itemsets: pd.DataFrame) -> pd.DataFrame:
    """Convert set-valued itemsets into CSV-friendly strings."""
    out = itemsets.copy()
    if "itemsets" in out.columns:
        out["itemsets"] = out["itemsets"].apply(format_itemset)
    return out


def exportable_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Convert rule sets into readable columns for export."""
    if rules is None or rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    out = rules.copy()
    if "antecedents" in out.columns:
        out["antecedents"] = out["antecedents"].apply(format_itemset)
    if "consequents" in out.columns:
        out["consequents"] = out["consequents"].apply(format_itemset)
    return out
