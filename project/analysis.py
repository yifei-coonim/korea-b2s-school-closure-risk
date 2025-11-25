# Project Title:
# Can B2S (births-to-seat mismatch) predict school closure risk,
# and is the predicted risk concentrated in vulnerable non-urban regions?

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# hide warnings for clean output
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (8, 5)

# input & output paths
DATA_FILE_RAW = "korea_primary_dataset_2020.2024.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# load data and compute derived B2S variables
def load_and_prepare_data(path_in: str) -> pd.DataFrame:
    df = pd.read_csv(path_in)

    # clean numeric columns (they may come with commas)
    numeric_cols = ["births", "grade1_students", "elem_school"]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # region type flag (urban=1, non-urban=0)
    if "region_code" in df.columns:
        df["region_code"] = pd.to_numeric(df["region_code"], errors="coerce").astype("Int64")

    if "region_type_flag" in df.columns:
        df["region_type_flag"] = pd.to_numeric(
            df["region_type_flag"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        if "region_type" in df.columns:
            # basic mapping for dataset
            df["region_type_flag"] = df["region_type"].map(
                {"CapitalArea": 1, "MetroCity": 1, "Province": 0}
            ).fillna(0).astype(int)
        else:
            df["region_type_flag"] = 0

    # core indicators
    df["births_per_school"] = df["births"] / df["elem_school"]
    df["grade1_per_school"] = df["grade1_students"] / df["elem_school"]
    df["b2s_ratio"] = df["grade1_students"] / df["births"]

    # standardize B2S (for regression)
    mean_b2s = df["b2s_ratio"].mean()
    std_b2s = df["b2s_ratio"].std(ddof=0)
    df["b2s_z"] = (df["b2s_ratio"] - mean_b2s) / std_b2s

    # define "low B2S" as bottom 20% (possible threshold zone)
    thresh_20 = df["b2s_ratio"].quantile(0.20)
    df["b2s_low_flag"] = (df["b2s_ratio"] <= thresh_20).astype(int)

    # compute next year's school count (for y target)
    df = df.sort_values(["region_name", "year"])
    df["elem_school_next"] = df.groupby("region_name", observed=True)["elem_school"].shift(-1)

    # school decrease indicator (y)
    mask_known_next = df["elem_school_next"].notna()
    df["y_school_decrease_next"] = np.where(
        mask_known_next & (df["elem_school_next"] < df["elem_school"]),
        1.0,
        0.0,
    )
    df.loc[~mask_known_next, "y_school_decrease_next"] = np.nan

    # optional interaction term used in fairness test
    df["b2s_regiontype_interaction"] = df["b2s_ratio"] * df["region_type_flag"]

    return df


# logistic regression with interaction (used in fairness and prediction)
def fit_logit_interaction(df: pd.DataFrame):
    cols = ["y_school_decrease_next", "b2s_z", "region_type_flag"]
    subset = df[cols].dropna()
    if subset["y_school_decrease_next"].nunique() < 2:
        return None, subset

    subset["interaction"] = subset["b2s_z"] * subset["region_type_flag"]
    y = subset["y_school_decrease_next"]
    X = sm.add_constant(subset[["b2s_z", "region_type_flag", "interaction"]])
    model = sm.Logit(y, X).fit(disp=False)
    return model, subset


# Analysis 1: structural differences across regions (B2S distribution)
def analyze_structural_difference(df: pd.DataFrame) -> None:
    print("\n=== Analysis 1: structural differences ===")
    max_year = df["year"].max()
    df_last = df[df["year"] == max_year].copy()

    # English region names for clean plots
    label_col = "region_en"
    df_last_sorted = df_last.sort_values("b2s_ratio", ascending=False)

    print("\n[Latest year B2S by region]")
    print(df_last_sorted[[label_col, "year", "b2s_ratio"]])

    fig, ax = plt.subplots()
    ax.bar(df_last_sorted[label_col].astype(str), df_last_sorted["b2s_ratio"])
    ax.set_xticklabels(df_last_sorted[label_col].astype(str), rotation=45, ha="right")
    ax.set_ylabel("B2S ratio (grade1_students / births)")
    ax.set_title(f"B2S structure by region in {int(max_year)}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "structural_b2s_2024_by_region.png")
    plt.savefig(path)
    plt.close(fig)


# Analysis 2: threshold / non-linear effect of B2S
def analyze_threshold_non_linearity(df: pd.DataFrame) -> None:
    print("\n=== Analysis 2: threshold / non-linearity ===")
    df_train = df[df["y_school_decrease_next"].notna()].copy()
    if df_train.empty:
        print("No rows with known next-year outcome.")
        return

    df_train["b2s_quantile3"] = pd.qcut(
        df_train["b2s_ratio"],
        q=[0, 1 / 3, 2 / 3, 1.0],
        labels=["Low", "Middle", "High"],
    )

    quantile_stats = (
        df_train.groupby("b2s_quantile3", observed=True)["y_school_decrease_next"]
        .mean()
        .to_frame(name="closure_rate")
        .sort_index()
    )

    print("\n[Closure rate by 3-level B2S quantile]")
    print(quantile_stats)

    fig, ax = plt.subplots()
    ax.plot(
        quantile_stats.index.astype(str),
        quantile_stats["closure_rate"],
        marker="o",
    )
    ax.set_xlabel("B2S level (3-quantile)")
    ax.set_ylabel("Next-year school decrease rate")
    ax.set_title("Closure risk by B2S level")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "threshold_b2s_quantile3_closure_rate.png")
    plt.savefig(path)
    plt.close(fig)


# Analysis 3: fairness test (urban vs non-urban)
def analyze_fairness(df: pd.DataFrame) -> None:
    print("\n=== Analysis 3: fairness across region types ===")
    df_train = df[df["y_school_decrease_next"].notna()].copy()

    pivot = (
        df_train.groupby(["region_type_flag", "b2s_low_flag"], observed=True)["y_school_decrease_next"]
        .mean()
        .reset_index()
    )

    print("\n[Closure rate by region_type_flag (0=non-urban, 1=urban) and B2S_low_flag]")
    print(pivot)

    fig, ax = plt.subplots()
    labels = ["B2S_not_low", "B2S_low"]
    x = np.arange(len(labels))
    width = 0.35

    # two groups: rural vs urban
    for flag, shift, color, legend_label in [
        (0, -width / 2, "tab:blue", "region_type_flag=0"),
        (1, +width / 2, "tab:orange", "region_type_flag=1"),
    ]:
        sub = pivot[pivot["region_type_flag"] == flag].sort_values("b2s_low_flag")
        if sub.empty:
            continue
        y_vals = sub["y_school_decrease_next"].values
        if len(y_vals) == 1:
            if sub["b2s_low_flag"].iloc[0] == 0:
                y_vals = np.array([y_vals[0], 0.0])
            else:
                y_vals = np.array([0.0, y_vals[0]])
        ax.bar(x + shift, y_vals, width, label=legend_label)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Next-year school decrease rate")
    ax.set_title("Closure risk: low vs non-low B2S, by region type")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fairness_low_b2s_by_region_type.png")
    plt.savefig(path)
    plt.close(fig)

    model_inter, subset = fit_logit_interaction(df_train)

    print("\n[Logistic regression: y ~ b2s_z + region_type_flag + interaction]")
    print(model_inter.summary())

    fig, ax = plt.subplots()
    x_grid = np.linspace(subset["b2s_z"].min(), subset["b2s_z"].max(), 200)

    # predicted lines for two region types
    for flag, color, legend_label in [
        (0, "tab:blue", "region_type_flag=0"),
        (1, "tab:orange", "region_type_flag=1"),
    ]:
        X_grid = pd.DataFrame(
            {
                "const": 1.0,
                "b2s_z": x_grid,
                "region_type_flag": flag,
                "interaction": x_grid * flag,
            }
        )
        p_pred = model_inter.predict(X_grid)
        ax.plot(x_grid, p_pred, color=color, label=legend_label)

    ax.set_xlabel("B2S z-score")
    ax.set_ylabel("Predicted probability of next-year school decrease")
    ax.set_title("Fitted closure risk by B2S and region type")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fairness_b2s_curve_by_region_type.png")
    plt.savefig(path)
    plt.close(fig)


# Analysis 4: prediction for next year (2025)
def analyze_prediction_next_year(df: pd.DataFrame) -> None:
    print("\n=== Analysis 4: prediction for next year by region ===")
    df_train = df[df["y_school_decrease_next"].notna()].copy()

    model_inter, _ = fit_logit_interaction(df_train)

    max_year = df["year"].max()
    df_pred = df[df["year"] == max_year].copy()

    X_pred = pd.DataFrame(
        {
            "const": 1.0,
            "b2s_z": df_pred["b2s_z"],
            "region_type_flag": df_pred["region_type_flag"],
            "interaction": df_pred["b2s_z"] * df_pred["region_type_flag"],
        }
    )
    df_pred["pred_risk_next_year"] = model_inter.predict(X_pred)

    label_col = "region_en"
    df_pred_sorted = df_pred.sort_values("pred_risk_next_year", ascending=False)

    print("\n[Predicted risk of next-year school decrease by region]")
    print(df_pred_sorted[[label_col, "year", "pred_risk_next_year"]])

    fig, ax = plt.subplots()
    ax.bar(
        df_pred_sorted[label_col].astype(str),
        df_pred_sorted["pred_risk_next_year"],
    )
    ax.set_xticklabels(df_pred_sorted[label_col].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Predicted probability of next-year school decrease")
    ax.set_title(f"Predicted closure risk in {int(max_year) + 1} by region")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "prediction_next_year_by_region.png")
    plt.savefig(path)
    plt.close(fig)


def main() -> None:
    print("=== B2S mismatch and school closure project ===")
    df = load_and_prepare_data(DATA_FILE_RAW)
    print(f"Data loaded: {len(df)} rows")
    analyze_structural_difference(df)
    analyze_threshold_non_linearity(df)
    analyze_fairness(df)
    analyze_prediction_next_year(df)
    print("\nAll analyses completed.")


if __name__ == "__main__":
    main()
