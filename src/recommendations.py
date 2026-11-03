"""Association-rule based product recommendations."""
from __future__ import annotations
import pandas as pd
from .mining import format_itemset


def recommend_products(
    selected_products: list[str] | tuple[str, ...] | set[str],
    rules: pd.DataFrame,
    top_n: int = 10,
    min_confidence: float = 0.0,
    min_lift: float = 0.0,
) -> pd.DataFrame:
    """Rank rule consequents whose entire antecedent is in the selected basket.

    Recommendations already present in the basket are excluded. If several rules
    recommend the same product, retain its strongest confidence/lift rule and
    include the number of selected items that triggered it.
    """
    columns = ["Product", "Confidence", "Lift", "Rule Support", "Matched Rule", "Matched Items"]
    selected = set(selected_products or [])
    if not selected or rules is None or rules.empty:
        return pd.DataFrame(columns=columns)
    candidates: dict[str, dict] = {}

    for _, row in rules.iterrows():
        antecedents = set(row.get("antecedents", []))
        consequents = set(row.get("consequents", []))
        if not antecedents or not antecedents.issubset(selected):
            continue
        if float(row.get("confidence", 0)) < min_confidence:
            continue
        if float(row.get("lift", 0)) < min_lift:
            continue
        for product in consequents - selected:
            record = {
                "Product": str(product),
                "Confidence": float(row.get("confidence", 0)),
                "Lift": float(row.get("lift", 0)),
                "Rule Support": float(row.get("support", 0)),
                "Matched Rule": f"{format_itemset(antecedents)} → {format_itemset(consequents)}",
                "Matched Items": len(antecedents),
            }
            old = candidates.get(str(product))
            if old is None or (record["Confidence"], record["Lift"], record["Rule Support"]) > (
                old["Confidence"], old["Lift"], old["Rule Support"]
            ):
                candidates[str(product)] = record

    if not candidates:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(candidates.values(), columns=columns)
    return result.sort_values(
        ["Confidence", "Lift", "Rule Support"], ascending=False
    ).head(max(1, int(top_n))).reset_index(drop=True)
