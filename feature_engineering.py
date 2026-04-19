"""Feature Engineering Pipeline Takes the cleaned datasets from data_cleaning.py and produces a
single DataFrame with all features and the target variable.

Pipeline:
  1. Load cleaned datasets
  2. Label target variable (repeat repair within 180 days & 10m)
  3. Join potholes -> road assets (spatial: point-in-polygon / nearest)
  4. Join potholes -> traffic (spatial: nearest intersection, max 500m)
  5. Join potholes -> weather (date match)
  6. Engineer final features
  7. Export dataset

Output: datasets/model_ready.csv
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# GLOBALS

DATASETS_DIR = Path("datasets")
OUTPUT_DIR = DATASETS_DIR

# Target variable: repair is "repeat" if another repair happens within this distance and time window
REPEAT_DISTANCE_M = 10
REPEAT_WINDOW_DAYS = 180

# Traffic join: max distance to assign a traffic station
TRAFFIC_MAX_DISTANCE_M = 500

# Projected CRS for Montreal (MTM Zone 8)
CRS_PROJECTED = "EPSG:32188"
CRS_GEO = "EPSG:4326"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ==================================
# LOAD CLEANED DATASETS
# ===================================


def load_datasets():
    """Load all cleaned datasets from data_cleaning.py outputs.

    Arguments:
        Takes no arguments, loads all cleaned datasets from the data_cleaning.py outputs.
    Returns:
        Returns dataframes and geodataframes for each dataset.
    """
    log.info("=== Loading cleaned datasets ===")
    # Load cleaned datasets
    potholes = gpd.read_file(DATASETS_DIR / "potholes_cleaned.gpkg")
    potholes["Date"] = pd.to_datetime(potholes["Date"])
    log.info(f"Potholes: {len(potholes):,}")

    road_assets = gpd.read_file(DATASETS_DIR / "road_assets_cleaned.gpkg")
    log.info(f"Road assets: {len(road_assets):,}")

    traffic = gpd.read_file(DATASETS_DIR / "traffic_cleaned.gpkg")
    log.info(f"Traffic stations: {len(traffic):,}")

    weather = pd.read_csv(DATASETS_DIR / "weather_cleaned.csv")
    weather["Date"] = pd.to_datetime(weather["Date"])
    log.info(f"Weather days: {len(weather):,}")
    # Return dataframes and geodataframes for each dataset
    return potholes, road_assets, traffic, weather


# =================================================
# TARGET VARIABLE
# ================================================


def label_repeat_repairs(potholes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Label each repair 1 if another repair occurs within REPEAT_DISTANCE_M and REPEAT_WINDOW_DAYS
    in the future, 0 otherwise.

    Uses a KD-tree on projected coordinates for efficient spatial lookup,
     then checks temporal proximity for nearby pairs.
    Arguments:
        Takes a GeoDataFrame of potholes
    Returns:
        New GeoDataFrame with column "is_repeat" indicating whether each repair is a repeat.
    """
    log.info("=== Labeling repeat repairs ===")

    # Project to meters
    potholes_proj = potholes.to_crs(CRS_PROJECTED)
    coords = np.column_stack([potholes_proj.geometry.x, potholes_proj.geometry.y])
    dates = potholes["Date"].values  # numpy datetime64

    # Build KDtree
    log.info("Building spatial index...")
    tree = cKDTree(coords)

    # For each point find all neighbors within REPEAT_DISTANCE_M
    log.info(f"Querying neighbors within {REPEAT_DISTANCE_M}m...")
    neighbor_pairs = tree.query_pairs(r=REPEAT_DISTANCE_M, output_type="ndarray")
    log.info(f"Spatial neighbor pairs found: {len(neighbor_pairs):,}")

    # Check temporal condition within REPEAT_WINDOW_DAYS but not same day
    is_repeat = np.zeros(len(potholes), dtype=int)

    for i, j in neighbor_pairs:
        diff_days = (dates[j] - dates[i]) / np.timedelta64(1, "D")

        # If j comes after i within the window → i is a repeat
        if 1 <= diff_days <= REPEAT_WINDOW_DAYS:
            is_repeat[i] = 1
        # If i comes after j within the window → j is a repeat
        if 1 <= -diff_days <= REPEAT_WINDOW_DAYS:
            is_repeat[j] = 1

    potholes = potholes.copy()
    potholes["is_repeat"] = is_repeat
    # Log repeat rate
    n_repeat = is_repeat.sum()
    pct = n_repeat / len(potholes) * 100
    log.info(f"Repeat repairs: {n_repeat:,} / {len(potholes):,} ({pct:.1f}%)")
    # Return the modified GeoDataFrame
    return potholes


# ======================================
# JOIN ROAD ASSET DATA
# ======================================


def join_road_assets(
    potholes: gpd.GeoDataFrame, road_assets: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Spatial join: assign each pothole the road segment it falls on.
    Arguments:
        Takes a GeoDataFrame of potholes and a GeoDataFrame of road assets.
    Returns:
        Returns joined GeoDataFrame.
    """
    log.info("=== Joining road assets ===")

    # Ensure same CRS
    potholes_proj = potholes.to_crs(CRS_PROJECTED)
    roads_proj = road_assets.to_crs(CRS_PROJECTED)

    # Buffer road lines by 15m and do intersection join
    roads_buffered = roads_proj.copy()
    roads_buffered["geometry"] = roads_proj.geometry.buffer(15)

    log.info("Spatial join (buffer intersection)...")
    joined = gpd.sjoin(potholes_proj, roads_buffered, how="left", predicate="within")

    # Some potholes may match multiple road segments so keep closest
    if "index_right" in joined.columns:
        # For duplicates keep first match
        joined = joined[~joined.index.duplicated(keep="first")]

    matched = joined["ID_VOI_VOIRIE_AGR"].notna().sum()
    unmatched = joined["ID_VOI_VOIRIE_AGR"].isna().sum()
    log.info(f"Buffer join: {matched:,} matched, {unmatched:,} unmatched")

    # For unmatched try nearest neighbor within 50m
    if unmatched > 0:
        log.info("Nearest-neighbor join for unmatched potholes...")
        unmatched_mask = joined["ID_VOI_VOIRIE_AGR"].isna()
        unmatched_pts = potholes_proj.loc[unmatched_mask.index[unmatched_mask]]

        if len(unmatched_pts) > 0 and len(roads_proj) > 0:
            nearest = gpd.sjoin_nearest(
                unmatched_pts[["geometry"]],
                roads_proj,
                how="left",
                max_distance=50,
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
    log.info(
        f"Total road asset match rate: {total_matched:,} / {len(joined):,} ({total_pct:.1f}%)"
    )

    # Restore original CRS and geometry
    result = potholes.copy()
    road_cols = [
        "ID_VOI_VOIRIE_AGR",
        "CATEGORIECHAUSSEE_REF",
        "DATECONSTRUCTION",
        "DATERESURFACAGE",
        "MATERIAUCHAUSSEE_REF",
        "TYPEFONDATION_REF",
        "LAST_SURFACE_DATE",
        "date_unknown",
    ]
    for col in road_cols:
        if col in joined.columns:
            result[col] = joined[col].values
    # Return joined GeoDataFrame
    return result


# ========================================
# TRAFFIC DATA JOIN
# ========================================


def join_traffic(
    potholes: gpd.GeoDataFrame, traffic: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Nearest-neighbor join: assign each pothole the traffic volume of the
     closest intersection within TRAFFIC_MAX_DISTANCE_M.
    Arguments:
        Takes a GeoDataFrame of potholes and a GeoDataFrame of traffic.
    Returns:
        Returns a joined GeoDataFrame.
    """
    log.info("=== Joining traffic ===")

    if len(traffic) == 0:
        log.warning("No traffic data - skipping")
        potholes = potholes.copy()
        potholes["avg_daily_traffic"] = np.nan
        potholes["traffic_distance_m"] = np.nan
        return potholes

    # Project to meters
    potholes_proj = potholes.to_crs(CRS_PROJECTED)
    traffic_proj = traffic.to_crs(CRS_PROJECTED)

    # Build KDtree on traffic coordinates
    traffic_coords = np.column_stack([traffic_proj.geometry.x, traffic_proj.geometry.y])
    pothole_coords = np.column_stack(
        [potholes_proj.geometry.x, potholes_proj.geometry.y]
    )

    tree = cKDTree(traffic_coords)
    distances, indices = tree.query(pothole_coords, k=1)

    potholes = potholes.copy()
    potholes["avg_daily_traffic"] = traffic.iloc[indices]["avg_daily_traffic"].values
    potholes["traffic_distance_m"] = distances

    # set to NaN if too far from any station
    too_far = potholes["traffic_distance_m"] > TRAFFIC_MAX_DISTANCE_M
    potholes.loc[too_far, "avg_daily_traffic"] = np.nan

    # log stats
    within_range = (~too_far).sum()
    pct = within_range / len(potholes) * 100
    log.info(
        f"Potholes within {TRAFFIC_MAX_DISTANCE_M}m of a traffic station: "
        f"{within_range:,} / {len(potholes):,} ({pct:.1f}%)"
    )
    log.info(f"Median join distance: {potholes['traffic_distance_m'].median():.0f}m")
    # return joined GeoDataFrame
    return potholes


# =======================================
# JOIN WEATHER DATA
# ========================================


def join_weather(potholes: gpd.GeoDataFrame, weather: pd.DataFrame) -> gpd.GeoDataFrame:
    """Exact date join: merge weather features onto each pothole by repair date.
    Arguments:
        Takes a GeoDataFrame of potholes and a DataFrame of weather.
    Returns:
        Returns a joined GeoDataFrame.
    """
    log.info("=== Joining weather ===")

    potholes = potholes.copy()

    # Normalize both dates to date-only for merge
    potholes["_merge_date"] = potholes["Date"].dt.normalize()
    weather["_merge_date"] = weather["Date"].dt.normalize()

    weather_cols = [
        "_merge_date",
        "MeanTemp",
        "MaxTemp",
        "MinTemp",
        "Precip",
        "SnowOnGround",
        "freeze_thaw",
        "precip_30d",
        "precip_60d",
        "freeze_thaw_30d",
        "freeze_thaw_60d",
        "below_zero",
        "days_since_freeze_thaw",
    ]
    weather_subset = weather[[c for c in weather_cols if c in weather.columns]]

    before = len(potholes)
    potholes = potholes.merge(weather_subset, on="_merge_date", how="left")
    potholes = potholes.drop(columns=["_merge_date"])

    matched = potholes["MeanTemp"].notna().sum()
    log.info(f"Weather match: {matched:,} / {before:,} ({matched/before*100:.1f}%)")

    return potholes


# =======================================
# FEATURE ENGINEERING
# ===========================================


def engineer_features(potholes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute final features from the joined data.

    Arguments:
        Takes final joined GeoDataFrame.
    Returns:
        Returns a GeoDataFrame including engineered features.
    """
    log.info("=== Engineering features ===")
    # Drop geometry column
    df = pd.DataFrame(potholes.drop(columns=["geometry"]))

    # Road surface and age features
    if "DATECONSTRUCTION" in df.columns:
        df["DATECONSTRUCTION"] = pd.to_datetime(df["DATECONSTRUCTION"], errors="coerce")
        df["road_age"] = (df["Date"] - df["DATECONSTRUCTION"]).dt.days / 365.25
        # Clip unreasonable ages
        df.loc[df["road_age"] < 0, "road_age"] = np.nan
        df.loc[df["road_age"] > 150, "road_age"] = np.nan

    if "LAST_SURFACE_DATE" in df.columns:
        df["LAST_SURFACE_DATE"] = pd.to_datetime(
            df["LAST_SURFACE_DATE"], errors="coerce"
        )
        df["years_since_surface"] = (
            df["Date"] - df["LAST_SURFACE_DATE"]
        ).dt.days / 365.25
        df.loc[df["years_since_surface"] < 0, "years_since_surface"] = np.nan
        df.loc[df["years_since_surface"] > 150, "years_since_surface"] = np.nan

    # Temporal features from repair date
    df["repair_month"] = df["Date"].dt.month
    df["repair_dow"] = df["Date"].dt.dayofweek  # 0=Mon, 6=Sun
    df["repair_year"] = df["Date"].dt.year

    # Encode equipment type if available
    if "Appareil" in df.columns:
        # Keep top N equipment types, group rest as 'Other'
        top_equip = df["Appareil"].value_counts().head(10).index
        df["equipment_type"] = df["Appareil"].where(
            df["Appareil"].isin(top_equip), other="Other"
        )
    else:
        df["equipment_type"] = "Unknown"

    # Encode road material if available
    if "MATERIAUCHAUSSEE_REF" in df.columns:
        df["road_material"] = df["MATERIAUCHAUSSEE_REF"].fillna("Unknown")
    else:
        df["road_material"] = "Unknown"

    # date unkonwn flag
    if "date_unknown" not in df.columns:
        df["date_unknown"] = 0

    # Final columns
    feature_cols = [
        # Target
        "is_repeat",
        # Road characteristics
        "road_age",
        "years_since_surface",
        "date_unknown",
        "road_material",
        "equipment_type",
        # Traffic
        "avg_daily_traffic",
        # Weather repair day
        "MeanTemp",
        "MaxTemp",
        "MinTemp",
        "Precip",
        "SnowOnGround",
        "below_zero",
        "freeze_thaw",
        # Weather rolling
        "precip_30d",
        "precip_60d",
        "freeze_thaw_30d",
        "freeze_thaw_60d",
        "days_since_freeze_thaw",
        # Temporal
        "repair_month",
        "repair_dow",
        "repair_year",
    ]

    # Only keep columns that actually exist
    available_features = [c for c in feature_cols if c in df.columns]
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        log.warning(f"Missing feature columns (will be excluded): {missing_features}")

    result = df[available_features].copy()

    # Summary Log
    log.info(f"Final dataset shape: {result.shape}")
    log.info(f"Target distribution:\n{result['is_repeat'].value_counts()}")
    log.info(f"Missing values per feature:\n{result.isna().sum()}")
    # Return final dataset
    return result


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    log.info("Starting feature engineering pipeline")

    # Load
    potholes, road_assets, traffic, weather = load_datasets()

    # Label target
    potholes = label_repeat_repairs(potholes)

    # Joins
    potholes = join_road_assets(potholes, road_assets)
    potholes = join_traffic(potholes, traffic)
    potholes = join_weather(potholes, weather)

    # Engineer features
    model_df = engineer_features(potholes)

    # Export final dataset
    out_path = OUTPUT_DIR / "model_ready.csv"
    model_df.to_csv(out_path, index=False)
    log.info(f"Saved model-ready dataset to {out_path}")

    log.info("=" * 60)
    log.info("Feature engineering complete!")


if __name__ == "__main__":
    main()
