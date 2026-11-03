"""Data validation and preprocessing for retail transactions."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import pandas as pd

REQUIRED_COLUMNS = ["InvoiceNo", "Description", "Quantity"]
OPTIONAL_COLUMNS = ["InvoiceDate", "UnitPrice", "CustomerID"]


class DataValidationError(ValueError):
    """Raised when uploaded transaction data cannot be processed."""


@dataclass
class CleaningStats:
    input_rows: int = 0
    output_rows: int = 0
    missing_invoice_rows: int = 0
    missing_product_rows: int = 0
    invalid_quantity_rows: int = 0
    cancelled_invoice_rows: int = 0
    duplicate_invoice_product_rows: int = 0
    removed_rows: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def validate_columns(df: pd.DataFrame) -> None:
    """Validate the minimum schema and reject empty input."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Input must be a CSV table.")
    if df.empty:
        raise DataValidationError("The uploaded file contains no data rows.")
    normalized_columns = [str(col).strip() for col in df.columns]
    missing = [col for col in REQUIRED_COLUMNS if col not in normalized_columns]
    if missing:
        raise DataValidationError(
            "Missing required column(s): " + ", ".join(missing)
            + ". Required columns are InvoiceNo, Description, Quantity."
        )


def preprocess_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningStats]:
    """Clean transaction lines and return a copy plus transparent row counts.

    Policy: remove missing invoice/product, non-numeric or non-positive quantity,
    cancelled invoices (InvoiceNo begins with C, case-insensitive), and duplicate
    invoice-product pairs. Duplicate quantities are not silently summed; the first
    valid row is retained to keep the policy predictable.
    """
    validate_columns(df)
    stats = CleaningStats(input_rows=len(df))
    work = df.copy()
    # Standardize column names' surrounding whitespace without changing spelling.
    work.columns = [str(c).strip() for c in work.columns]
    # Revalidate in case a header had surrounding whitespace.
    validate_columns(work)

    invoice = work["InvoiceNo"].astype("string").str.strip()
    description = work["Description"].astype("string").str.strip()
    stats.missing_invoice_rows = int(invoice.isna().sum() + invoice.eq("").fillna(False).sum())
    stats.missing_product_rows = int(description.isna().sum() + description.eq("").fillna(False).sum())

    valid = invoice.notna() & invoice.ne("").fillna(False)
    work = work.loc[valid].copy()
    invoice = work["InvoiceNo"].astype("string").str.strip()
    cancelled = invoice.str.upper().str.startswith("C", na=False)
    stats.cancelled_invoice_rows = int(cancelled.sum())
    work = work.loc[~cancelled].copy()

    description = work["Description"].astype("string").str.strip()
    valid_product = description.notna() & description.ne("").fillna(False)
    work = work.loc[valid_product].copy()

    qty = pd.to_numeric(work["Quantity"], errors="coerce")
    invalid_qty = qty.isna() | (qty <= 0)
    stats.invalid_quantity_rows = int(invalid_qty.sum())
    work = work.loc[~invalid_qty].copy()
    work["Quantity"] = qty.loc[~invalid_qty].astype(float)

    work["InvoiceNo"] = work["InvoiceNo"].astype(str).str.strip()
    work["Description"] = (
        work["Description"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    )

    # Optional date and price columns are parsed gently; invalid values become missing.
    if "InvoiceDate" in work.columns:
        work["InvoiceDate"] = pd.to_datetime(work["InvoiceDate"], errors="coerce")
    if "UnitPrice" in work.columns:
        work["UnitPrice"] = pd.to_numeric(work["UnitPrice"], errors="coerce")
    if "CustomerID" in work.columns:
        work["CustomerID"] = work["CustomerID"].astype("string").str.strip()

    before_dedupe = len(work)
    work = work.drop_duplicates(subset=["InvoiceNo", "Description"], keep="first")
    stats.duplicate_invoice_product_rows = int(before_dedupe - len(work))
    work = work.reset_index(drop=True)
    stats.output_rows = len(work)
    stats.removed_rows = max(0, stats.input_rows - stats.output_rows)
    return work, stats


def make_basket_matrix(cleaned: pd.DataFrame, max_transactions: int = 20_000,
                       max_products: int = 1_000) -> pd.DataFrame:
    """Create a boolean invoice-by-product matrix with size safeguards."""
    validate_columns(cleaned)
    if cleaned.empty:
        return pd.DataFrame()
    unique_invoices = cleaned["InvoiceNo"].nunique()
    unique_products = cleaned["Description"].nunique()
    if unique_invoices > max_transactions:
        raise DataValidationError(
            f"Dataset has {unique_invoices:,} invoices. Limit is {max_transactions:,}. "
            "Filter to a smaller subset before mining."
        )
    if unique_products > max_products:
        raise DataValidationError(
            f"Dataset has {unique_products:,} products. Limit is {max_products:,}. "
            "Filter to frequently sold products or use a smaller subset."
        )
    basket = pd.crosstab(cleaned["InvoiceNo"], cleaned["Description"]).gt(0)
    basket.index.name = "InvoiceNo"
    return basket.astype(bool)
