"""
Data Cleaning Pipeline for Montreal Pothole Repair Failure Prediction
=====================================================================
Cleans and exports each dataset independently. No joins or feature
engineering happen here — that's for the next stage.

Datasets:
  1. Potholes  → datasets/potholes_cleaned.gpkg
  2. Road Assets  → datasets/road_assets_cleaned.gpkg
  3. Road Condition → datasets/road_condition_cleaned.csv
  4. Traffic  → datasets/traffic_cleaned.gpkg
  5. Weather  → datasets/weather_cleaned.csv

Improvements over original notebook:
  - Coordinate bounds filtering for potholes (Montreal bbox)
  - Same-day/same-location deduplication for potholes
  - Keeps equipment type (Appareil) as potential feature
  - Consistent road asset filtering (vehicle roads only)
  - Flags missing date coverage in road assets
  - Multi-year average traffic per intersection (more stable)
  - Weather rolling features computed here for easy downstream merge
  - Logging throughout for auditability
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

# ── Config ──────────────────────────────────────────────────────────────────

DATASETS_DIR = Path("datasets")
OUTPUT_DIR = DATASETS_DIR  # cleaned files go in same parent dir

# Montreal bounding box (generous)
MTL_LAT_MIN, MTL_LAT_MAX = 45.40, 45.72
MTL_LON_MIN, MTL_LON_MAX = -73.98, -73.47

# Deduplication: repairs within this distance (meters) and time (days)
# on the same day are considered duplicates
DEDUP_DISTANCE_M = 5
DEDUP_SAME_DAY = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── 1. Potholes ─────────────────────────────────────────────────────────────

def clean_potholes() -> gpd.GeoDataFrame:
    """Load, clean, deduplicate, and bounds-filter pothole repair records."""
    log.info("=== Cleaning potholes ===")

    # --- Load all years ---
    csv_years = range(2016, 2021)  # 2016-2020 are CSVs
    gpkg_years = range(2021, 2026)  # 2021-2025 are GPKGs

    gdfs = []

    for year in csv_years:
        path = DATASETS_DIR / f"pothole_fixes/potholes_{year}.csv"
        if not path.exists():
            log.warning(f"Missing: {path}")
            continue
        df = pd.read_csv(path)
        geometry = gpd.points_from_xy(df["Longitude"], df["Latitude"])
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        gdf["source_year"] = year
        gdfs.append(gdf)

    for year in gpkg_years:
        path = DATASETS_DIR / f"pothole_fixes/potholes_{year}.gpkg"
        if not path.exists():
            log.warning(f"Missing: {path}")
            continue
        gdf = gpd.read_file(path)
        gdf["source_year"] = year
        gdfs.append(gdf)

    # Normalize CRS before concatenation
    gdfs = [gdf.to_crs("EPSG:4326") for gdf in gdfs]
    potholes = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    log.info(f"Raw records loaded: {len(potholes):,}")

    # --- Consolidate date column ---
    # Some years use 'DateJour', others use 'Date' or 'DateHeure'
    if "DateJour" in potholes.columns:
        potholes["DateJour"] = potholes["DateJour"].fillna(
            potholes["Date"].astype(str).str[:10]
        )
    else:
        potholes["DateJour"] = potholes["Date"].astype(str).str[:10]

    potholes["Date"] = pd.to_datetime(potholes["DateJour"], errors="coerce")

    # --- Consolidate lat/lon from geometry where missing ---
    potholes["Latitude"] = potholes["Latitude"].fillna(potholes.geometry.y)
    potholes["Longitude"] = potholes["Longitude"].fillna(potholes.geometry.x)

    # --- Drop columns we don't need ---
    # Keep 'Appareil' (equipment type) — could be a useful feature
    drop_cols = [c for c in ["Véhicule", "DateHeure", "DateJour"] if c in potholes.columns]
    potholes = potholes.drop(columns=drop_cols)

    # --- Filter: remove rows with no valid date ---
    before = len(potholes)
    potholes = potholes.dropna(subset=["Date"])
    log.info(f"Dropped {before - len(potholes):,} rows with missing dates")

    # --- Filter: Montreal bounding box ---
    before = len(potholes)
    in_bounds = (
        (potholes["Latitude"] >= MTL_LAT_MIN)
        & (potholes["Latitude"] <= MTL_LAT_MAX)
        & (potholes["Longitude"] >= MTL_LON_MIN)
        & (potholes["Longitude"] <= MTL_LON_MAX)
    )
    potholes = potholes[in_bounds].copy()
    log.info(f"Removed {before - len(potholes):,} out-of-bounds records")

    # --- Deduplicate: same date, very close location ---
    # Project to MTM Zone 8 for meter-based distance, round to DEDUP_DISTANCE_M grid
    before = len(potholes)
    potholes_proj = potholes.to_crs("EPSG:32188")
    potholes["_x_round"] = (potholes_proj.geometry.x / DEDUP_DISTANCE_M).round()
    potholes["_y_round"] = (potholes_proj.geometry.y / DEDUP_DISTANCE_M).round()
    potholes["_date_str"] = potholes["Date"].dt.date.astype(str)

    potholes = potholes.drop_duplicates(subset=["_date_str", "_x_round", "_y_round"], keep="first")
    potholes = potholes.drop(columns=["_x_round", "_y_round", "_date_str"])
    log.info(f"Removed {before - len(potholes):,} duplicate repairs (same day, within {DEDUP_DISTANCE_M}m)")

    # --- Summary ---
    log.info(f"Final pothole records: {len(potholes):,}")
    log.info(f"Date range: {potholes['Date'].min().date()} to {potholes['Date'].max().date()}")
    if "Appareil" in potholes.columns:
        log.info(f"Equipment types: {potholes['Appareil'].nunique()} unique values")

    # --- Export ---
    out_path = OUTPUT_DIR / "potholes_cleaned.gpkg"
    potholes.to_file(out_path, driver="GPKG")
    log.info(f"Saved to {out_path}")

    return potholes


# ── 2. Road Assets ──────────────────────────────────────────────────────────

def clean_road_assets() -> gpd.GeoDataFrame:
    """Load road asset data, filter to vehicle roads, parse dates."""
    log.info("=== Cleaning road assets ===")

    road_assets = gpd.read_file(DATASETS_DIR / "road_assets/voirie_actif.geojson")
    log.info(f"Raw road asset records: {len(road_assets):,}")

    # --- Keep relevant columns ---
    keep_cols = [
        "ID_VOI_VOIRIE_AGR",
        "CATEGORIECHAUSSEE_REF",
        "DATECONSTRUCTION",
        "DATERESURFACAGE",
        "MATERIAUCHAUSSEE_REF",
        "TYPEFONDATION_REF",
        "UTILISATION_REF",
        "geometry",
    ]
    available = [c for c in keep_cols if c in road_assets.columns]
    road_assets = road_assets[available]

    # --- Filter to vehicle roads only ---
    road_assets = road_assets[road_assets["UTILISATION_REF"] == "Véhicule"].copy()
    log.info(f"Vehicle roads: {len(road_assets):,}")

    # --- Parse dates ---
    for col in ["DATECONSTRUCTION", "DATERESURFACAGE"]:
        if col in road_assets.columns:
            road_assets[col] = pd.to_datetime(
                road_assets[col].astype(str).str[:8],
                format="%Y%m%d",
                errors="coerce",
            )

    # --- Derived: last surface date + missing date flag ---
    road_assets["LAST_SURFACE_DATE"] = road_assets["DATERESURFACAGE"].fillna(
        road_assets["DATECONSTRUCTION"]
    )

    n_missing = road_assets["LAST_SURFACE_DATE"].isna().sum()
    pct_missing = n_missing / len(road_assets) * 100
    road_assets["date_unknown"] = road_assets["LAST_SURFACE_DATE"].isna().astype(int)
    log.info(f"Roads with no construction or resurfacing date: {n_missing:,} ({pct_missing:.1f}%)")

    # --- Export ---
    out_path = OUTPUT_DIR / "road_assets_cleaned.gpkg"
    road_assets.to_file(out_path, driver="GPKG")
    log.info(f"Saved to {out_path}")

    return road_assets


# ── 3. Road Condition ───────────────────────────────────────────────────────

def clean_road_condition() -> pd.DataFrame:
    """Load and concatenate road condition assessments from multiple years.

    Key improvement: keeps DateReleve so downstream code can do
    temporal matching (use the most recent assessment BEFORE each repair).
    """
    log.info("=== Cleaning road condition ===")

    # Only keep years that actually have PCI/IRI scores (2020+).
    # 2010, 2015, 2018 have zero PCI and zero IRI — they add no usable signal.
    files_and_years = [
        ("road_condition/auscultation-chaussees-2020-arteriel.csv", 2020),
        ("road_condition/auscultation-chaussees-2022-local.csv", 2022),
        ("road_condition/auscultation-chaussee-2024.csv", 2024),
    ]

    dfs = []
    for filename, year in files_and_years:
        path = DATASETS_DIR / filename
        if not path.exists():
            log.warning(f"Missing: {path}")
            continue
        df = pd.read_csv(path)
        df["source_year"] = year
        dfs.append(df)
        log.info(f"  {year}: {len(df):,} records")

    road_condition = pd.concat(dfs, ignore_index=True)

    # --- Keep relevant columns ---
    keep_cols = [
        "ID_TRC", "Rue", "De", "A", "Longueur", "Arrondissement",
        "DateReleve", "Indice_PCI", "Indice_IRI", "source_year",
    ]
    available = [c for c in keep_cols if c in road_condition.columns]
    road_condition = road_condition[available]

    # --- Parse date ---
    road_condition["DateReleve"] = pd.to_datetime(
        road_condition["DateReleve"], format="mixed", errors="coerce"
    )

    # --- Flag coverage ---
    log.info(f"Total condition records: {len(road_condition):,}")
    log.info(f"Unique road segments (ID_TRC): {road_condition['ID_TRC'].nunique():,}")

    pci_coverage = road_condition["Indice_PCI"].notna().sum()
    iri_coverage = road_condition["Indice_IRI"].notna().sum()
    log.info(f"PCI non-null: {pci_coverage:,}, IRI non-null: {iri_coverage:,}")

    by_year = road_condition.groupby("source_year").agg(
        records=("ID_TRC", "count"),
        pci_available=("Indice_PCI", lambda x: x.notna().sum()),
        iri_available=("Indice_IRI", lambda x: x.notna().sum()),
    )
    log.info(f"Coverage by year:\n{by_year}")

    # --- Export ---
    out_path = OUTPUT_DIR / "road_condition_cleaned.csv"
    road_condition.to_csv(out_path, index=False)
    log.info(f"Saved to {out_path}")

    return road_condition


# ── 4. Traffic ──────────────────────────────────────────────────────────────

def clean_traffic() -> gpd.GeoDataFrame:
    """Load traffic counting data and compute stable per-intersection averages.

    Improvement: computes a multi-year average per intersection instead of
    yearly averages, since counting campaigns are sparse and inconsistent.
    Also computes a yearly version for temporal matching if desired.
    """
    log.info("=== Cleaning traffic ===")

    traffic_files = list((DATASETS_DIR / "traffic").glob("*.csv"))
    if not traffic_files:
        log.warning("No traffic CSV files found!")
        return gpd.GeoDataFrame()

    traffic_raw = pd.concat(
        [pd.read_csv(f) for f in traffic_files], ignore_index=True
    )
    log.info(f"Raw traffic records: {len(traffic_raw):,}")

    # --- Filter to vehicle types ---
    vehicle_types = [
        "Autos", "Camions", "Camions legers", "Camions Lourds",
        "Camions porteurs", "Camions articules", "Bus", "Motos",
    ]
    traffic_veh = traffic_raw[
        traffic_raw["Description_Code_Banque"].isin(vehicle_types)
    ].copy()
    log.info(f"Vehicle records: {len(traffic_veh):,}")

    # --- Sum movement columns for total volume per row ---
    movement_cols = [
        "NBLT", "NBT", "NBRT", "SBLT", "SBT", "SBRT",
        "EBLT", "EBT", "EBRT", "WBLT", "WBT", "WBRT",
    ]
    traffic_veh["total_volume"] = traffic_veh[movement_cols].sum(axis=1)

    # --- Aggregate by intersection and date ---
    traffic_veh["Date"] = pd.to_datetime(traffic_veh["Date"], errors="coerce")

    traffic_daily = traffic_veh.groupby(["Id_Intersection", "Date"]).agg(
        Nom_Intersection=("Nom_Intersection", "first"),
        Longitude=("Longitude", "first"),
        Latitude=("Latitude", "first"),
        total_volume=("total_volume", "sum"),
    ).reset_index()

    # --- Multi-year average per intersection (stable estimate) ---
    traffic_avg = traffic_daily.groupby("Id_Intersection").agg(
        Nom_Intersection=("Nom_Intersection", "first"),
        Longitude=("Longitude", "first"),
        Latitude=("Latitude", "first"),
        avg_daily_traffic=("total_volume", "mean"),
        count_days=("total_volume", "count"),
        min_date=("Date", "min"),
        max_date=("Date", "max"),
    ).reset_index()

    log.info(f"Intersections with traffic data: {len(traffic_avg):,}")
    log.info(f"Median counting days per intersection: {traffic_avg['count_days'].median():.0f}")

    # --- Also save yearly version for optional temporal matching ---
    traffic_daily["Year"] = traffic_daily["Date"].dt.year
    traffic_yearly = traffic_daily.groupby(["Id_Intersection", "Year"]).agg(
        Nom_Intersection=("Nom_Intersection", "first"),
        Longitude=("Longitude", "first"),
        Latitude=("Latitude", "first"),
        avg_daily_traffic=("total_volume", "mean"),
    ).reset_index()

    # --- Convert both to GeoDataFrame ---
    traffic_geo = gpd.GeoDataFrame(
        traffic_avg,
        geometry=gpd.points_from_xy(traffic_avg["Longitude"], traffic_avg["Latitude"]),
        crs="EPSG:4326",
    )

    traffic_yearly_geo = gpd.GeoDataFrame(
        traffic_yearly,
        geometry=gpd.points_from_xy(traffic_yearly["Longitude"], traffic_yearly["Latitude"]),
        crs="EPSG:4326",
    )

    # --- Export both ---
    out_path = OUTPUT_DIR / "traffic_cleaned.gpkg"
    traffic_geo.to_file(out_path, driver="GPKG")
    log.info(f"Saved multi-year averages to {out_path}")

    out_path_yearly = OUTPUT_DIR / "traffic_yearly_cleaned.gpkg"
    traffic_yearly_geo.to_file(out_path_yearly, driver="GPKG")
    log.info(f"Saved yearly averages to {out_path_yearly}")

    return traffic_geo


# ── 5. Weather ──────────────────────────────────────────────────────────────

def clean_weather() -> pd.DataFrame:
    """Load and clean weather data. Compute rolling features and repair-day features.

    Improvements:
      - Rolling windows (30d/60d) for precipitation and freeze-thaw
      - Repair-day features: temp at repair, precip on repair day,
        was it freezing, days since last freeze-thaw event
      - All computed here so downstream merge is a simple date-join
    """
    log.info("=== Cleaning weather ===")

    weather_files = sorted((DATASETS_DIR / "weather").glob("montreal_weather_*.csv"))
    if not weather_files:
        log.warning("No weather files found!")
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in weather_files]
    weather = pd.concat(dfs, ignore_index=True)
    log.info(f"Raw weather records: {len(weather):,}")

    # --- Keep and rename columns ---
    col_map = {
        "Date/Time": "Date",
        "Max Temp (°C)": "MaxTemp",
        "Min Temp (°C)": "MinTemp",
        "Mean Temp (°C)": "MeanTemp",
        "Total Precip (mm)": "Precip",
        "Snow on Grnd (cm)": "SnowOnGround",
    }
    available = {k: v for k, v in col_map.items() if k in weather.columns}
    weather = weather[list(available.keys())].rename(columns=available)
    weather["Date"] = pd.to_datetime(weather["Date"], errors="coerce")
    weather = weather.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Drop duplicate dates (if multiple stations or re-entries)
    weather = weather.drop_duplicates(subset=["Date"], keep="first")

    # --- Fill small gaps in numeric columns ---
    numeric_cols = ["MaxTemp", "MinTemp", "MeanTemp", "Precip", "SnowOnGround"]
    for col in numeric_cols:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    # SnowOnGround: nulls in warm months (May-Oct) are just 0, not missing data.
    # Only interpolate actual winter gaps.
    if "SnowOnGround" in weather.columns:
        month = weather["Date"].dt.month
        summer_mask = month.between(5, 10)
        snow_null = weather["SnowOnGround"].isna()

        summer_filled = (summer_mask & snow_null).sum()
        weather.loc[summer_mask & snow_null, "SnowOnGround"] = 0.0
        log.info(f"  SnowOnGround: set {summer_filled} summer nulls to 0")

        winter_gaps = weather["SnowOnGround"].isna().sum()
        if winter_gaps > 0:
            log.info(f"  SnowOnGround: interpolating {winter_gaps} remaining winter gaps")
            weather["SnowOnGround"] = weather["SnowOnGround"].interpolate(
                method="linear", limit=7
            )

    # Interpolate small gaps in other columns (temp, precip)
    for col in ["MaxTemp", "MinTemp", "MeanTemp", "Precip"]:
        if col in weather.columns:
            gap_count = weather[col].isna().sum()
            if gap_count > 0:
                log.info(f"  {col}: filling {gap_count} missing values via interpolation")
                weather[col] = weather[col].interpolate(method="linear", limit=7)

    # --- Freeze-thaw: day where max > 0 and min < 0 ---
    weather["freeze_thaw"] = (
        (weather["MaxTemp"] > 0) & (weather["MinTemp"] < 0)
    ).astype(int)

    # --- Rolling features (30d, 60d) ---
    weather = weather.set_index("Date").sort_index()

    weather["precip_30d"] = weather["Precip"].rolling("30D", min_periods=1).sum()
    weather["precip_60d"] = weather["Precip"].rolling("60D", min_periods=1).sum()
    weather["freeze_thaw_30d"] = weather["freeze_thaw"].rolling("30D", min_periods=1).sum()
    weather["freeze_thaw_60d"] = weather["freeze_thaw"].rolling("60D", min_periods=1).sum()

    # --- Repair-day features ---
    # Was it freezing on this day?
    weather["below_zero"] = (weather["MeanTemp"] < 0).astype(int)

    # Days since last freeze-thaw event
    ft_dates = weather.index[weather["freeze_thaw"] == 1]
    weather["days_since_freeze_thaw"] = np.nan
    if len(ft_dates) > 0:
        # For each day, find the most recent freeze-thaw date
        ft_series = pd.Series(ft_dates, index=ft_dates)
        weather["_last_ft"] = ft_series.reindex(weather.index, method="ffill")
        weather["days_since_freeze_thaw"] = (
            weather.index - weather["_last_ft"]
        ).dt.days
        weather = weather.drop(columns=["_last_ft"])

    weather = weather.reset_index()

    # --- Summary ---
    log.info(f"Weather date range: {weather['Date'].min().date()} to {weather['Date'].max().date()}")
    log.info(f"Total freeze-thaw days: {weather['freeze_thaw'].sum():,}")
    log.info(f"Feature columns: {list(weather.columns)}")

    # --- Export ---
    out_path = OUTPUT_DIR / "weather_cleaned.csv"
    weather.to_csv(out_path, index=False)
    log.info(f"Saved to {out_path}")

    return weather


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("Starting data cleaning pipeline")
    log.info(f"Datasets directory: {DATASETS_DIR.resolve()}")

    potholes = clean_potholes()
    road_assets = clean_road_assets()
    road_condition = clean_road_condition()
    traffic = clean_traffic()
    weather = clean_weather()

    log.info("=" * 60)
    log.info("Pipeline complete. Cleaned files:")
    log.info(f"  Potholes:       {len(potholes):>10,} records")
    log.info(f"  Road assets:    {len(road_assets):>10,} records")
    log.info(f"  Road condition: {len(road_condition):>10,} records")
    log.info(f"  Traffic:        {len(traffic):>10,} records")
    log.info(f"  Weather:        {len(weather):>10,} records")


if __name__ == "__main__":
    main()