"""
Feature Engineering Pipeline for Montreal Pothole Repair Failure Prediction
===========================================================================
Takes the cleaned datasets from data_cleaning.py and produces a single
model-ready DataFrame with all features and the target variable.

Pipeline:
  1. Load cleaned datasets
  2. Label target variable (repeat repair within 180 days & 10m)
  3. Join potholes → road assets (spatial: point-in-polygon / nearest)
  4. Join potholes → road condition (temporal: most recent assessment before repair)
  5. Join potholes → traffic (spatial: nearest intersection, max 500m)
  6. Join potholes → weather (date match)
  7. Engineer final features
  8. Export model-ready dataset

Output: datasets/model_ready.csv
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ── Config ──────────────────────────────────────────────────────────────────

DATASETS_DIR = Path("datasets")
OUTPUT_DIR = DATASETS_DIR

# Target variable: repair is "repeat" if another repair happens
# within this distance and time window
REPEAT_DISTANCE_M = 10
REPEAT_WINDOW_DAYS = 180

# Traffic join: max distance to assign a traffic station
TRAFFIC_MAX_DISTANCE_M = 500

# Projected CRS for Montreal (MTM Zone 8) — meter-based distances
CRS_PROJECTED = "EPSG:32188"
CRS_GEO = "EPSG:4326"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Load cleaned datasets ──────────────────────────────────────────────────

def load_datasets():
    """Load all cleaned datasets from data_cleaning.py outputs."""
    log.info("=== Loading cleaned datasets ===")

    potholes = gpd.read_file(DATASETS_DIR / "potholes_cleaned.gpkg")
    potholes["Date"] = pd.to_datetime(potholes["Date"])
    log.info(f"Potholes: {len(potholes):,}")

    road_assets = gpd.read_file(DATASETS_DIR / "road_assets_cleaned.gpkg")
    log.info(f"Road assets: {len(road_assets):,}")

    road_condition = pd.read_csv(DATASETS_DIR / "road_condition_cleaned.csv")
    road_condition["DateReleve"] = pd.to_datetime(road_condition["DateReleve"])
    log.info(f"Road condition: {len(road_condition):,}")

    traffic = gpd.read_file(DATASETS_DIR / "traffic_cleaned.gpkg")
    log.info(f"Traffic stations: {len(traffic):,}")

    weather = pd.read_csv(DATASETS_DIR / "weather_cleaned.csv")
    weather["Date"] = pd.to_datetime(weather["Date"])
    log.info(f"Weather days: {len(weather):,}")

    return potholes, road_assets, road_condition, traffic, weather


# ── 1. Target Variable ─────────────────────────────────────────────────────

def label_repeat_repairs(potholes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Label each repair: 1 if another repair occurs within REPEAT_DISTANCE_M
    and REPEAT_WINDOW_DAYS in the future, 0 otherwise.

    Uses a KD-tree on projected coordinates for efficient spatial lookup,
    then checks temporal proximity for nearby pairs.
    """
    log.info("=== Labeling repeat repairs ===")

    # Project to meters
    potholes_proj = potholes.to_crs(CRS_PROJECTED)
    coords = np.column_stack([potholes_proj.geometry.x, potholes_proj.geometry.y])
    dates = potholes["Date"].values  # numpy datetime64

    # Build KD-tree
    log.info("Building spatial index...")
    tree = cKDTree(coords)

    # For each point, find all neighbors within REPEAT_DISTANCE_M
    log.info(f"Querying neighbors within {REPEAT_DISTANCE_M}m...")
    neighbor_pairs = tree.query_pairs(r=REPEAT_DISTANCE_M, output_type="ndarray")
    log.info(f"Spatial neighbor pairs found: {len(neighbor_pairs):,}")

    # Check temporal condition: the later repair must be within 1-180 days
    # of the earlier one (not same-day, which would be the repair itself)
    is_repeat = np.zeros(len(potholes), dtype=int)

    for i, j in neighbor_pairs:
        diff_days = (dates[j] - dates[i]) / np.timedelta64(1, "D")

        # If j comes after i within the window → i is a repeat (it gets re-repaired)
        if 1 <= diff_days <= REPEAT_WINDOW_DAYS:
            is_repeat[i] = 1
        # If i comes after j within the window → j is a repeat
        if 1 <= -diff_days <= REPEAT_WINDOW_DAYS:
            is_repeat[j] = 1

    potholes = potholes.copy()
    potholes["is_repeat"] = is_repeat

    n_repeat = is_repeat.sum()
    pct = n_repeat / len(potholes) * 100
    log.info(f"Repeat repairs: {n_repeat:,} / {len(potholes):,} ({pct:.1f}%)")

    return potholes


# ── 2. Road Asset Join ──────────────────────────────────────────────────────

def join_road_assets(
    potholes: gpd.GeoDataFrame, road_assets: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Spatial join: assign each pothole the road segment it falls on.

    Strategy:
      1. Try sjoin (point-in-polygon with small buffer on lines)
      2. For unmatched, use nearest-neighbor join
    """
    log.info("=== Joining road assets ===")

    # Ensure same CRS (projected for spatial accuracy)
    potholes_proj = potholes.to_crs(CRS_PROJECTED)
    roads_proj = road_assets.to_crs(CRS_PROJECTED)

    # Buffer road lines by 15m and do intersection join
    roads_buffered = roads_proj.copy()
    roads_buffered["geometry"] = roads_proj.geometry.buffer(15)

    log.info("Spatial join (buffer intersection)...")
    joined = gpd.sjoin(potholes_proj, roads_buffered, how="left", predicate="within")

    # Some potholes may match multiple road segments — keep closest
    if "index_right" in joined.columns:
        # For duplicates, we'll just keep the first match for now
        joined = joined[~joined.index.duplicated(keep="first")]

    matched = joined["ID_VOI_VOIRIE_AGR"].notna().sum()
    unmatched = joined["ID_VOI_VOIRIE_AGR"].isna().sum()
    log.info(f"Buffer join: {matched:,} matched, {unmatched:,} unmatched")

    # For unmatched, try nearest neighbor
    if unmatched > 0:
        log.info("Nearest-neighbor join for unmatched potholes...")
        unmatched_mask = joined["ID_VOI_VOIRIE_AGR"].isna()
        unmatched_pts = potholes_proj.loc[unmatched_mask.index[unmatched_mask]]

        if len(unmatched_pts) > 0 and len(roads_proj) > 0:
            nearest = gpd.sjoin_nearest(
                unmatched_pts[["geometry"]],
                roads_proj,
                how="left",
                max_distance=50,  # max 50m for nearest match
            )
            nearest = nearest[~nearest.index.duplicated(keep="first")]

            # Fill in the missing values
            road_cols = [c for c in road_assets.columns if c != "geometry"]
            for col in road_cols:
                if col in nearest.columns and col in joined.columns:
                    joined.loc[nearest.index, col] = nearest[col].values

            nn_matched = nearest["ID_VOI_VOIRIE_AGR"].notna().sum()
            log.info(f"Nearest-neighbor: {nn_matched:,} additional matches")

    total_matched = joined["ID_VOI_VOIRIE_AGR"].notna().sum()
    total_pct = total_matched / len(joined) * 100
    log.info(f"Total road asset match rate: {total_matched:,} / {len(joined):,} ({total_pct:.1f}%)")

    # Restore original CRS and geometry
    result = potholes.copy()
    road_cols = [
        "ID_VOI_VOIRIE_AGR", "CATEGORIECHAUSSEE_REF", "DATECONSTRUCTION",
        "DATERESURFACAGE", "MATERIAUCHAUSSEE_REF", "TYPEFONDATION_REF",
        "LAST_SURFACE_DATE", "date_unknown",
    ]
    for col in road_cols:
        if col in joined.columns:
            result[col] = joined[col].values

    return result


# ── 3. Road Condition Join ──────────────────────────────────────────────────

def join_road_condition(
    potholes: gpd.GeoDataFrame, road_condition: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Temporal join: for each pothole, find the most recent road condition
    assessment BEFORE the repair date, matched by road segment ID (ID_TRC).

    Since potholes don't have ID_TRC directly, we need to go through
    the geobase or use spatial matching. Here we use the Arrondissement
    and street name as a proxy, plus temporal matching.

    If ID_TRC is available from the road asset join, we use that directly.
    """
    log.info("=== Joining road condition ===")

    # Road condition has ID_TRC. We need to match potholes to road segments.
    # Strategy: spatial join potholes to the geobase (road network) to get ID_TRC,
    # then temporal-match to condition data.
    #
    # If the geobase isn't available, we fall back to a cruder approach:
    # for each pothole, find condition records in the same arrondissement
    # and pick the most recent one before the repair date.
    #
    # For now, let's try loading the geobase for ID_TRC matching.

    geobase_path = DATASETS_DIR / "road_network"
    geobase_file = None
    for ext in ["*.geojson", "*.gpkg", "*.shp"]:
        files = list(geobase_path.glob(ext)) if geobase_path.exists() else []
        if files:
            geobase_file = files[0]
            break

    # Also check datasets root
    if geobase_file is None:
        for ext in ["geobase.geojson", "geobase.gpkg", "geobase*.geojson"]:
            files = list(DATASETS_DIR.glob(ext))
            if files:
                geobase_file = files[0]
                break

    if geobase_file is not None:
        log.info(f"Loading geobase from {geobase_file} for ID_TRC matching...")
        potholes = _join_condition_via_geobase(potholes, road_condition, geobase_file)
    else:
        log.warning("No geobase found — using spatial proximity fallback for road condition")
        potholes = _join_condition_spatial_fallback(potholes, road_condition)

    has_pci = potholes["Indice_PCI"].notna().sum()
    has_iri = potholes["Indice_IRI"].notna().sum()
    log.info(f"Potholes with PCI: {has_pci:,} ({has_pci/len(potholes)*100:.1f}%)")
    log.info(f"Potholes with IRI: {has_iri:,} ({has_iri/len(potholes)*100:.1f}%)")

    return potholes


def _join_condition_via_geobase(
    potholes: gpd.GeoDataFrame,
    road_condition: pd.DataFrame,
    geobase_path: Path,
) -> gpd.GeoDataFrame:
    """Match potholes → geobase (for ID_TRC) → road condition (temporal)."""

    geobase = gpd.read_file(geobase_path)
    log.info(f"Geobase segments: {len(geobase):,}")

    # Find the ID_TRC column (might be named differently)
    id_col = None
    for candidate in ["ID_TRC", "id_trc", "ID_TRC_GEOBASE"]:
        if candidate in geobase.columns:
            id_col = candidate
            break

    if id_col is None:
        log.warning(f"No ID_TRC column found in geobase. Columns: {list(geobase.columns)[:20]}")
        return _join_condition_spatial_fallback(potholes, road_condition)

    # Spatial join: potholes → geobase (nearest road segment)
    potholes_proj = potholes.to_crs(CRS_PROJECTED)
    geobase_proj = geobase[[id_col, "geometry"]].to_crs(CRS_PROJECTED)

    log.info("Spatial join potholes → geobase for ID_TRC...")
    matched = gpd.sjoin_nearest(
        potholes_proj[["geometry"]],
        geobase_proj,
        how="left",
        max_distance=30,
    )
    matched = matched[~matched.index.duplicated(keep="first")]
    potholes = potholes.copy()
    potholes["ID_TRC"] = matched[id_col].values

    trc_matched = potholes["ID_TRC"].notna().sum()
    log.info(f"Potholes matched to geobase: {trc_matched:,} / {len(potholes):,}")

    # Now temporal join: for each pothole's ID_TRC and Date,
    # find the most recent condition assessment BEFORE that date
    log.info("Temporal matching: most recent condition before each repair...")
    potholes = _temporal_merge_condition(potholes, road_condition)

    return potholes


def _temporal_merge_condition(
    potholes: gpd.GeoDataFrame, road_condition: pd.DataFrame
) -> gpd.GeoDataFrame:
    """For each pothole with an ID_TRC, find the most recent road condition
    record for that segment with DateReleve <= pothole Date."""

    # Only work with potholes that have ID_TRC
    has_trc = potholes["ID_TRC"].notna()
    log.info(f"Potholes with ID_TRC for temporal matching: {has_trc.sum():,}")

    if has_trc.sum() == 0:
        potholes["Indice_PCI"] = np.nan
        potholes["Indice_IRI"] = np.nan
        potholes["condition_age_days"] = np.nan
        potholes["has_condition_score"] = 0
        return potholes

    # Prepare condition data: keep only rows with at least one score
    rc = road_condition[
        road_condition["Indice_PCI"].notna() | road_condition["Indice_IRI"].notna()
    ].copy()
    rc = rc.sort_values("DateReleve")

    # Prepare pothole subset
    ph = potholes.loc[has_trc, ["ID_TRC", "Date"]].copy()
    ph = ph.rename(columns={"Date": "repair_date"})
    ph["_idx"] = ph.index  # preserve original index

    # Merge on ID_TRC
    merged = ph.merge(
        rc[["ID_TRC", "DateReleve", "Indice_PCI", "Indice_IRI"]],
        on="ID_TRC",
        how="left",
    )

    # Keep only assessments BEFORE the repair
    merged = merged[merged["DateReleve"] <= merged["repair_date"]]

    # For each pothole, keep the most recent assessment
    merged = merged.sort_values("DateReleve").groupby("_idx").last().reset_index()

    # Compute how stale the condition data is
    merged["condition_age_days"] = (
        merged["repair_date"] - merged["DateReleve"]
    ).dt.days

    # Map back to potholes
    potholes = potholes.copy()
    potholes["Indice_PCI"] = np.nan
    potholes["Indice_IRI"] = np.nan
    potholes["condition_age_days"] = np.nan

    if len(merged) > 0:
        potholes.loc[merged["_idx"].values, "Indice_PCI"] = merged["Indice_PCI"].values
        potholes.loc[merged["_idx"].values, "Indice_IRI"] = merged["Indice_IRI"].values
        potholes.loc[merged["_idx"].values, "condition_age_days"] = merged["condition_age_days"].values

    potholes["has_condition_score"] = potholes["Indice_PCI"].notna().astype(int)

    return potholes


def _join_condition_spatial_fallback(
    potholes: gpd.GeoDataFrame, road_condition: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Fallback: if no geobase available, we can't reliably match potholes
    to road condition segments. Set condition features to NaN with flag."""
    log.warning("Road condition join skipped — no reliable spatial key available")
    potholes = potholes.copy()
    potholes["Indice_PCI"] = np.nan
    potholes["Indice_IRI"] = np.nan
    potholes["condition_age_days"] = np.nan
    potholes["has_condition_score"] = 0
    potholes["ID_TRC"] = np.nan
    return potholes


# ── 4. Traffic Join ─────────────────────────────────────────────────────────

def join_traffic(
    potholes: gpd.GeoDataFrame, traffic: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Nearest-neighbor join: assign each pothole the traffic volume
    of the closest intersection, but only if within TRAFFIC_MAX_DISTANCE_M."""
    log.info("=== Joining traffic ===")

    if len(traffic) == 0:
        log.warning("No traffic data — skipping")
        potholes = potholes.copy()
        potholes["avg_daily_traffic"] = np.nan
        potholes["traffic_distance_m"] = np.nan
        return potholes

    potholes_proj = potholes.to_crs(CRS_PROJECTED)
    traffic_proj = traffic.to_crs(CRS_PROJECTED)

    # Build KD-tree on traffic station coordinates
    traffic_coords = np.column_stack([
        traffic_proj.geometry.x, traffic_proj.geometry.y
    ])
    pothole_coords = np.column_stack([
        potholes_proj.geometry.x, potholes_proj.geometry.y
    ])

    tree = cKDTree(traffic_coords)
    distances, indices = tree.query(pothole_coords, k=1)

    potholes = potholes.copy()
    potholes["avg_daily_traffic"] = traffic.iloc[indices]["avg_daily_traffic"].values
    potholes["traffic_distance_m"] = distances

    # Cap: set to NaN if too far from any station
    too_far = potholes["traffic_distance_m"] > TRAFFIC_MAX_DISTANCE_M
    potholes.loc[too_far, "avg_daily_traffic"] = np.nan

    within_range = (~too_far).sum()
    pct = within_range / len(potholes) * 100
    log.info(f"Potholes within {TRAFFIC_MAX_DISTANCE_M}m of a traffic station: "
             f"{within_range:,} / {len(potholes):,} ({pct:.1f}%)")
    log.info(f"Median join distance: {potholes['traffic_distance_m'].median():.0f}m")

    return potholes


# ── 5. Weather Join ─────────────────────────────────────────────────────────

def join_weather(
    potholes: gpd.GeoDataFrame, weather: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Exact date join: merge weather features onto each pothole by repair date."""
    log.info("=== Joining weather ===")

    potholes = potholes.copy()

    # Normalize both dates to date-only for clean merge
    potholes["_merge_date"] = potholes["Date"].dt.normalize()
    weather["_merge_date"] = weather["Date"].dt.normalize()

    weather_cols = [
        "_merge_date", "MeanTemp", "MaxTemp", "MinTemp", "Precip",
        "SnowOnGround", "freeze_thaw", "precip_30d", "precip_60d",
        "freeze_thaw_30d", "freeze_thaw_60d", "below_zero",
        "days_since_freeze_thaw",
    ]
    weather_subset = weather[[c for c in weather_cols if c in weather.columns]]

    before = len(potholes)
    potholes = potholes.merge(weather_subset, on="_merge_date", how="left")
    potholes = potholes.drop(columns=["_merge_date"])

    matched = potholes["MeanTemp"].notna().sum()
    log.info(f"Weather match: {matched:,} / {before:,} ({matched/before*100:.1f}%)")

    return potholes


# ── 6. Feature Engineering ──────────────────────────────────────────────────

def engineer_features(potholes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute final features from the joined data."""
    log.info("=== Engineering features ===")

    df = pd.DataFrame(potholes.drop(columns=["geometry"]))

    # --- Road age and surface age ---
    if "DATECONSTRUCTION" in df.columns:
        df["DATECONSTRUCTION"] = pd.to_datetime(df["DATECONSTRUCTION"], errors="coerce")
        df["road_age"] = (df["Date"] - df["DATECONSTRUCTION"]).dt.days / 365.25
        # Clip unreasonable ages
        df.loc[df["road_age"] < 0, "road_age"] = np.nan
        df.loc[df["road_age"] > 150, "road_age"] = np.nan

    if "LAST_SURFACE_DATE" in df.columns:
        df["LAST_SURFACE_DATE"] = pd.to_datetime(df["LAST_SURFACE_DATE"], errors="coerce")
        df["years_since_surface"] = (df["Date"] - df["LAST_SURFACE_DATE"]).dt.days / 365.25
        df.loc[df["years_since_surface"] < 0, "years_since_surface"] = np.nan
        df.loc[df["years_since_surface"] > 150, "years_since_surface"] = np.nan

    # --- Temporal features from repair date ---
    df["repair_month"] = df["Date"].dt.month
    df["repair_dow"] = df["Date"].dt.dayofweek  # 0=Mon, 6=Sun
    df["repair_year"] = df["Date"].dt.year

    # --- Encode equipment type if available ---
    if "Appareil" in df.columns:
        # Keep top N equipment types, group rest as 'Other'
        top_equip = df["Appareil"].value_counts().head(10).index
        df["equipment_type"] = df["Appareil"].where(
            df["Appareil"].isin(top_equip), other="Other"
        )
    else:
        df["equipment_type"] = "Unknown"

    # --- Encode road material if available ---
    if "MATERIAUCHAUSSEE_REF" in df.columns:
        df["road_material"] = df["MATERIAUCHAUSSEE_REF"].fillna("Unknown")
    else:
        df["road_material"] = "Unknown"

    # --- date_unknown flag (from road assets) ---
    if "date_unknown" not in df.columns:
        df["date_unknown"] = 0

    # --- has_condition_score flag ---
    if "has_condition_score" not in df.columns:
        df["has_condition_score"] = df["Indice_PCI"].notna().astype(int)

    # --- Select final columns ---
    feature_cols = [
        # Target
        "is_repeat",
        # Road characteristics
        "road_age", "years_since_surface", "date_unknown",
        "road_material", "equipment_type",
        # Road condition (with staleness indicator)
        "Indice_PCI", "Indice_IRI", "condition_age_days", "has_condition_score",
        # Traffic
        "avg_daily_traffic",
        # Weather — repair day
        "MeanTemp", "MaxTemp", "MinTemp", "Precip", "SnowOnGround",
        "below_zero", "freeze_thaw",
        # Weather — rolling
        "precip_30d", "precip_60d", "freeze_thaw_30d", "freeze_thaw_60d",
        "days_since_freeze_thaw",
        # Temporal
        "repair_month", "repair_dow", "repair_year",
    ]

    # Only keep columns that actually exist
    available_features = [c for c in feature_cols if c in df.columns]
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        log.warning(f"Missing feature columns (will be excluded): {missing_features}")

    result = df[available_features].copy()

    # --- Summary ---
    log.info(f"Final dataset shape: {result.shape}")
    log.info(f"Target distribution:\n{result['is_repeat'].value_counts()}")
    log.info(f"Missing values per feature:\n{result.isna().sum()}")

    return result


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("Starting feature engineering pipeline")

    # Load
    potholes, road_assets, road_condition, traffic, weather = load_datasets()

    # Label target
    potholes = label_repeat_repairs(potholes)

    # Joins
    potholes = join_road_assets(potholes, road_assets)
    potholes = join_road_condition(potholes, road_condition)
    potholes = join_traffic(potholes, traffic)
    potholes = join_weather(potholes, weather)

    # Engineer features
    model_df = engineer_features(potholes)

    # Export
    out_path = OUTPUT_DIR / "model_ready.csv"
    model_df.to_csv(out_path, index=False)
    log.info(f"Saved model-ready dataset to {out_path}")

    log.info("=" * 60)
    log.info("Feature engineering complete!")


if __name__ == "__main__":
    main()