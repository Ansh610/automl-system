import pandas as pd
import numpy as np


def generate_insights(df):

    insights = []

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    target = None

    for col in df.columns:

        if col.lower() in [
            "target",
            "label",
            "converted",
            "conversion",
            "class",
            "output",
            "y",
        ]:

            target = col
            break

    if target is None:
        return [
            "No target column detected for generating insights."
        ]

    conversion_rate = round(df[target].mean() * 100, 2)

    insights.append(
        f"Overall conversion rate is {conversion_rate}%."
    )

    for col in numeric_cols:

        if col == target:
            continue

        try:

            median = df[col].median()

            high = df[df[col] >= median][target].mean()

            low = df[df[col] < median][target].mean()

            if pd.isna(high) or pd.isna(low):
                continue

            if high > low:

                ratio = round(high / (low + 1e-8), 2)

                insights.append(
                    f"Higher '{col}' values increase conversion by {ratio}x."
                )

            elif low > high:

                ratio = round(low / (high + 1e-8), 2)

                insights.append(
                    f"Lower '{col}' values increase conversion by {ratio}x."
                )

        except Exception:
            continue

    missing = int(df.isnull().sum().sum())

    insights.append(
        f"Dataset contains {missing} missing values."
    )

    insights.append(
        f"Dataset has {len(df)} rows and {len(df.columns)} columns."
    )

    return insights