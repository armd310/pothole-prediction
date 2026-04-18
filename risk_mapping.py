"""
Risk Mapping for Montreal Pothole Repair Failure Prediction
===========================================================
Takes the trained model and spatial data to produce:
  1. Road-segment-level risk map (predicted repeat-repair probability)
  2. Arrondissement-level aggregated risk (if meaningful variation exists)

Outputs:
  - Static matplotlib figures (.png)
  - Interactive Folium maps (.html)
  - All saved to datasets/model_results/

Requires:
  - potholes_cleaned.gpkg (geometry)
  - model_ready.csv (features)
  - road_assets_cleaned.gpkg (road segments)
  - arrondissement.geojson (boundaries)
  - Trained model from model_training.py
"""

import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

try:
    import folium
    from folium.plugins import HeatMap

    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    print("WARNING: folium not installed. Run: pip install folium")
    print("Interactive maps will be skipped.")

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ──────────────────────────────────────────────────────────────────

DATASETS_DIR = Path("datasets")
RESULTS_DIR = DATASETS_DIR / "model_results"
RESULTS_DIR.mkdir(exist_ok=True)

ARRONDISSEMENT_PATH = DATASETS_DIR / "road_network" / "arrondissement.geojson"

CRS_PROJECTED = "EPSG:32188"
CRS_GEO = "EPSG:4326"

RANDOM_STATE = 42
TEST_SIZE = 0.2
FN_COST_RATIO = 3

# Arrondissement variation threshold: if std of mean risk across
# arrondissements is below this, skip the aggregated map
ARRONDISSEMENT_STD_THRESHOLD = 0.02

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Load & Retrain ──────────────────────────────────────────────────────────


def load_data_and_train():
    """Load model_ready.csv, retrain the best model (Random Forest), and return predictions with
    geometry attached."""
    log.info("=== Loading data and training model ===")

    # Load features
    df = pd.read_csv(DATASETS_DIR / "model_ready.csv")
    log.info(f"Model-ready data: {len(df):,} rows")

    # Load geometry from cleaned potholes
    potholes_geo = gpd.read_file(DATASETS_DIR / "potholes_cleaned.gpkg")
    potholes_geo["Date"] = pd.to_datetime(potholes_geo["Date"])
    log.info(f"Potholes with geometry: {len(potholes_geo):,}")

    # They should be the same length and order — verify
    if len(df) != len(potholes_geo):
        log.warning(
            f"Row count mismatch: model_ready={len(df)}, potholes_geo={len(potholes_geo)}"
        )
        log.warning("Using minimum length — rows should align by index")
        min_len = min(len(df), len(potholes_geo))
        df = df.iloc[:min_len]
        potholes_geo = potholes_geo.iloc[:min_len]

    # Prepare features and target
    y = df["is_repeat"]
    X = df.drop(columns=["is_repeat"])

    categorical_cols = [
        c
        for c in X.columns
        if X[c].dtype == "object" or c in ["equipment_type", "road_material"]
    ]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Train/test split (same seed as model_training.py for consistency)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Train Random Forest (best model from comparison)
    log.info("Training Random Forest...")
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

    pipe = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_leaf=20,
                    class_weight={0: 1, 1: FN_COST_RATIO},
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)

    # Predict on ALL data (we want a risk map of everything)
    log.info("Generating predictions for all potholes...")
    proba = pipe.predict_proba(X)[:, 1]

    # Attach predictions and geometry
    result = potholes_geo[["geometry", "Date"]].copy()
    result["risk_score"] = proba
    result["is_repeat"] = y.values
    result["Latitude"] = result.geometry.y
    result["Longitude"] = result.geometry.x

    log.info(f"Risk score range: {proba.min():.3f} – {proba.max():.3f}")
    log.info(f"Risk score mean: {proba.mean():.3f}, median: {np.median(proba):.3f}")

    return result, pipe


def _build_preprocessor(numeric_cols, categorical_cols):
    """Build preprocessor matching model_training.py."""
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="constant", fill_value="Unknown"),
                        ),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


# ── Arrondissement Aggregation ──────────────────────────────────────────────


def aggregate_by_arrondissement(pothole_risks: gpd.GeoDataFrame):
    """Spatial join potholes into arrondissement polygons and compute aggregate risk statistics.

    Returns None if variation is too low.
    """
    log.info("=== Arrondissement aggregation ===")

    if not ARRONDISSEMENT_PATH.exists():
        log.warning(f"Arrondissement file not found: {ARRONDISSEMENT_PATH}")
        return None

    arrond = gpd.read_file(ARRONDISSEMENT_PATH)
    arrond = arrond.to_crs(CRS_GEO)
    log.info(f"Arrondissement polygons: {len(arrond)}")
    log.info(f"Arrondissement columns: {list(arrond.columns)}")

    # Find the name column
    name_col = None
    for candidate in ["NOM", "Nom", "nom", "NAME", "name", "NOM_QR", "NOM_ARROND"]:
        if candidate in arrond.columns:
            name_col = candidate
            break

    if name_col is None:
        # Use first string column
        str_cols = [
            c for c in arrond.columns if arrond[c].dtype == "object" and c != "geometry"
        ]
        if str_cols:
            name_col = str_cols[0]
            log.info(f"Using '{name_col}' as arrondissement name column")
        else:
            log.warning("No name column found in arrondissement data")
            name_col = None

    # Spatial join: assign each pothole to an arrondissement
    pothole_pts = pothole_risks[["geometry", "risk_score", "is_repeat"]].copy()
    joined = gpd.sjoin(pothole_pts, arrond, how="left", predicate="within")

    if name_col and name_col in joined.columns:
        joined["arrond_name"] = joined[name_col]
    else:
        joined["arrond_name"] = joined["index_right"].astype(str)

    # Aggregate
    agg = (
        joined.groupby("arrond_name")
        .agg(
            mean_risk=("risk_score", "mean"),
            median_risk=("risk_score", "median"),
            repair_count=("risk_score", "count"),
            actual_repeat_rate=("is_repeat", "mean"),
            high_risk_count=("risk_score", lambda x: (x > 0.6).sum()),
        )
        .reset_index()
    )

    agg["high_risk_pct"] = agg["high_risk_count"] / agg["repair_count"] * 100

    # Check if there's meaningful variation
    risk_std = agg["mean_risk"].std()
    log.info(f"Arrondissement mean risk std: {risk_std:.4f}")
    log.info(f"Risk range: {agg['mean_risk'].min():.3f} – {agg['mean_risk'].max():.3f}")

    if risk_std < ARRONDISSEMENT_STD_THRESHOLD:
        log.info("Variation too low — skipping arrondissement map")
        return None

    log.info(f"Meaningful variation detected — including arrondissement map")
    log.info(
        f"\n{agg.sort_values('mean_risk', ascending=False).to_string(index=False)}"
    )

    # Merge back to polygons for mapping
    arrond_risk = arrond.merge(
        agg,
        left_on=name_col if name_col else arrond.index.name,
        right_on="arrond_name",
        how="left",
    )

    # Save stats
    agg.to_csv(RESULTS_DIR / "arrondissement_risk_stats.csv", index=False)
    log.info("Saved arrondissement_risk_stats.csv")

    return arrond_risk


# ── Static Maps ─────────────────────────────────────────────────────────────


def plot_static_risk_map(pothole_risks: gpd.GeoDataFrame, arrond_risk=None):
    """Generate static matplotlib risk maps."""
    log.info("=== Generating static maps ===")

    # ── Road-segment level: scatter of pothole risk scores ──
    fig, ax = plt.subplots(figsize=(14, 14))

    # Plot arrondissement boundaries as background if available
    if arrond_risk is not None:
        arrond_risk.boundary.plot(ax=ax, color="#cccccc", linewidth=0.5)

    # Sample for performance if very large
    if len(pothole_risks) > 100_000:
        plot_data = pothole_risks.sample(100_000, random_state=RANDOM_STATE)
        log.info("Sampling 100K points for static plot performance")
    else:
        plot_data = pothole_risks

    scatter = ax.scatter(
        plot_data.geometry.x,
        plot_data.geometry.y,
        c=plot_data["risk_score"],
        cmap="RdYlGn_r",  # red = high risk, green = low risk
        s=0.3,
        alpha=0.4,
        vmin=0,
        vmax=1,
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Predicted Repeat-Repair Probability", fontsize=12)

    ax.set_title(
        "Montreal Pothole Repair Risk Map\n(Predicted Repeat-Repair Probability)",
        fontsize=15,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    plt.tight_layout()

    out_path = RESULTS_DIR / "risk_map_points.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info(f"Saved {out_path}")

    # ── Arrondissement choropleth ──
    if arrond_risk is not None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Mean risk
        arrond_risk.plot(
            column="mean_risk",
            cmap="RdYlGn_r",
            legend=True,
            legend_kwds={"label": "Mean Predicted Risk", "shrink": 0.6},
            ax=axes[0],
            edgecolor="white",
            linewidth=0.8,
            missing_kwds={"color": "#f0f0f0", "label": "No data"},
        )
        axes[0].set_title("Mean Predicted Risk by Arrondissement", fontsize=13)
        axes[0].set_axis_off()

        # High risk percentage
        arrond_risk.plot(
            column="high_risk_pct",
            cmap="OrRd",
            legend=True,
            legend_kwds={"label": "% High Risk Repairs (>0.6)", "shrink": 0.6},
            ax=axes[1],
            edgecolor="white",
            linewidth=0.8,
            missing_kwds={"color": "#f0f0f0", "label": "No data"},
        )
        axes[1].set_title("High-Risk Repair Rate by Arrondissement", fontsize=13)
        axes[1].set_axis_off()

        plt.tight_layout()
        out_path = RESULTS_DIR / "risk_map_arrondissement.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        log.info(f"Saved {out_path}")


# ── Interactive Maps ────────────────────────────────────────────────────────


def plot_interactive_risk_map(pothole_risks: gpd.GeoDataFrame, arrond_risk=None):
    """Generate interactive Folium maps."""
    if not HAS_FOLIUM:
        log.warning("Folium not available — skipping interactive maps")
        return

    log.info("=== Generating interactive maps ===")

    # Montreal center
    center = [45.55, -73.65]

    # ── Heatmap weighted by risk score ──
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    # Sample for performance
    if len(pothole_risks) > 50_000:
        sample = pothole_risks.sample(50_000, random_state=RANDOM_STATE)
    else:
        sample = pothole_risks

    heat_data = [
        [row.geometry.y, row.geometry.x, row.risk_score] for _, row in sample.iterrows()
    ]

    HeatMap(
        heat_data,
        min_opacity=0.3,
        radius=8,
        blur=10,
        gradient={
            0.2: "green",
            0.4: "yellow",
            0.6: "orange",
            0.8: "red",
            1.0: "darkred",
        },
    ).add_to(m)

    # Add arrondissement boundaries if available
    if arrond_risk is not None:
        _add_arrondissement_layer(m, arrond_risk)

    out_path = RESULTS_DIR / "risk_map_heatmap.html"
    m.save(str(out_path))
    log.info(f"Saved {out_path}")

    # ── Choropleth map (arrondissement level) ──
    if arrond_risk is not None:
        m2 = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
        _add_arrondissement_layer(m2, arrond_risk, show_tooltip=True)

        out_path = RESULTS_DIR / "risk_map_arrondissement.html"
        m2.save(str(out_path))
        log.info(f"Saved {out_path}")

    # ── High-risk point map (only points with risk > 0.7) ──
    high_risk = pothole_risks[pothole_risks["risk_score"] > 0.7]
    log.info(f"High-risk points (>0.7): {len(high_risk):,}")

    if len(high_risk) > 0:
        m3 = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

        if arrond_risk is not None:
            arrond_boundary = arrond_risk.to_crs(CRS_GEO).copy()
            for col in arrond_boundary.columns:
                if pd.api.types.is_datetime64_any_dtype(arrond_boundary[col]):
                    arrond_boundary[col] = arrond_boundary[col].astype(str)
            folium.GeoJson(
                arrond_boundary.__geo_interface__,
                style_function=lambda x: {
                    "fillColor": "transparent",
                    "color": "#666",
                    "weight": 1,
                },
            ).add_to(m3)

        # Sample if too many
        if len(high_risk) > 10_000:
            high_risk_sample = high_risk.sample(10_000, random_state=RANDOM_STATE)
        else:
            high_risk_sample = high_risk

        for _, row in high_risk_sample.iterrows():
            color = _risk_color(row["risk_score"])
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=2,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=f"Risk: {row['risk_score']:.2f}",
            ).add_to(m3)

        out_path = RESULTS_DIR / "risk_map_high_risk_points.html"
        m3.save(str(out_path))
        log.info(f"Saved {out_path}")


def _add_arrondissement_layer(m, arrond_risk, show_tooltip=False):
    """Add choropleth arrondissement layer to a Folium map."""

    def style_function(feature):
        risk = feature["properties"].get("mean_risk")
        if risk is None or pd.isna(risk):
            return {
                "fillColor": "#f0f0f0",
                "color": "#999",
                "weight": 1,
                "fillOpacity": 0.3,
            }
        color = _risk_color(risk)
        return {"fillColor": color, "color": "white", "weight": 1.5, "fillOpacity": 0.6}

    geo_data = arrond_risk.to_crs(CRS_GEO).copy()

    # Convert any datetime columns to strings (Folium can't serialize Timestamps)
    for col in geo_data.columns:
        if pd.api.types.is_datetime64_any_dtype(geo_data[col]):
            geo_data[col] = geo_data[col].astype(str)

    if show_tooltip:
        tooltip = folium.GeoJsonTooltip(
            fields=["arrond_name", "mean_risk", "repair_count", "high_risk_pct"],
            aliases=["Arrondissement", "Mean Risk", "Total Repairs", "% High Risk"],
            localize=True,
        )
    else:
        tooltip = None

    folium.GeoJson(
        geo_data.__geo_interface__,
        style_function=style_function,
        tooltip=tooltip,
    ).add_to(m)


def _risk_color(risk):
    """Map a 0-1 risk score to a color."""
    if risk > 0.7:
        return "#d32f2f"  # dark red
    elif risk > 0.6:
        return "#f44336"  # red
    elif risk > 0.5:
        return "#ff9800"  # orange
    elif risk > 0.4:
        return "#ffc107"  # amber
    elif risk > 0.3:
        return "#cddc39"  # lime
    else:
        return "#4caf50"  # green


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    log.info("Starting risk mapping pipeline")

    # Load data and train model
    pothole_risks, model = load_data_and_train()

    # Arrondissement aggregation (check if meaningful)
    arrond_risk = aggregate_by_arrondissement(pothole_risks)

    if arrond_risk is not None:
        log.info("Arrondissement variation is meaningful — generating both map types")
    else:
        log.info("Arrondissement variation too low — generating point maps only")

    # Static maps
    plot_static_risk_map(pothole_risks, arrond_risk)

    # Interactive maps
    plot_interactive_risk_map(pothole_risks, arrond_risk)

    log.info("=" * 60)
    log.info("Risk mapping complete! Files saved to:")
    log.info(f"  {RESULTS_DIR}")
    for f in sorted(RESULTS_DIR.glob("risk_map*")):
        log.info(f"  - {f.name}")


if __name__ == "__main__":
    main()
