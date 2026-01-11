import pandas as pd
import numpy as np


def dataframe_summary(df: pd.DataFrame) -> dict:
    """
    Erstellt eine kompakte statistische Zusammenfassung eines Pandas DataFrames.

    Diese Funktion liefert die wichtigsten strukturellen und statistischen
    Informationen, um ein Dataset schnell zu verstehen oder an ein LLM
    weiterzugeben.

    Enthalten sind:
    - Anzahl Zeilen & Spalten
    - Spaltentypen
    - Fehlwerte
    - Numerische Kennzahlen (Mean, Std, Min, Max)
    - Kategorische Top-Werte

    Parameters
    ----------
    df : pd.DataFrame
        Das zu analysierende DataFrame.

    Returns
    -------
    dict
        Strukturierte Zusammenfassung des DataFrames.
    pd.DataFrame
        Tabellarische Zusammenfassung der Spaltenstatistiken.
    """

    summary = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "columns": {},
    }

    for col in df.columns:
        col_data = df[col]
        col_summary = {
            "dtype": str(col_data.dtype),
            "missing": int(col_data.isna().sum()),
            "missing_pct": float(col_data.isna().mean() * 100),
            "unique": int(col_data.nunique()),
        }

        # Numerische Spalten
        if pd.api.types.is_numeric_dtype(col_data):
            col_summary.update({
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
            })

        # Kategorische Spalten
        else:
            top_values = col_data.value_counts().head(3).to_dict()
            col_summary["top_values"] = top_values

        summary["columns"][col] = col_summary

    return summary, pd.DataFrame(summary["columns"]).T
