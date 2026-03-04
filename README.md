# Montreal Pothole Recurrence Prediction

A machine learning model to predict pothole recurrence within 6 months, enabling proactive infrastructure maintenance and resource allocation for municipal road services.

## Project Overview

This project uses historical pothole repair data, road condition metrics, and weather patterns to predict whether a repaired pothole will require re-repair within 6 months. The model helps optimize maintenance schedules and identify high-risk road segments for preventive intervention.

## Problem Statement

Pothole repairs are costly and disruptive. Many potholes recur within months of repair, indicating underlying road degradation. This model identifies patterns that predict recurrence, allowing cities to:
- Prioritize roads needing deeper repairs vs. quick patches
- Allocate maintenance budgets more effectively
- Schedule preventive maintenance before potholes reappear

## Technologies Used

- **Python** - Core programming language
- **scikit-learn** - Random Forest Classifier, model evaluation
- **pandas** - Data manipulation and cleaning
- **GeoPandas** - Geospatial data processing and spatial joins
- **NumPy** - Numerical operations
- **Shapely** - Geometric operations for spatial data
- **SciPy** - Spatial indexing with cKDTree for efficient nearest-neighbor searches
- **Jupyter Notebook** - Interactive development and analysis

## Data Sources

The model integrates three key data sources:
1. **Montreal Open Data** - Pothole repair records (location, date, repair type)
2. **Montreal Road Condition Data** - Road age, surface type, pavement condition indices (PCI, IRI)
3. **Environment Canada Weather Data** - Temperature, precipitation, freeze-thaw cycles, snow cover

**Dataset size:** ~700,000 pothole repair records

## 🔍 Features

The model uses 10 engineered features:

**Road Condition Metrics:**
- `road_age` - Age of road since construction
- `years_since_surface` - Years since last resurfacing
- `Indice_PCI` - Pavement Condition Index (0-100 scale)
- `Indice_IRI` - International Roughness Index (ride quality)

**Weather Impact Metrics:**
- `freeze_thaw_30d` - Number of freeze-thaw cycles in 30 days before repair
- `freeze_thaw_60d` - Number of freeze-thaw cycles in 60 days before repair
- `precip_30d` - Total precipitation (mm) in 30 days before repair
- `precip_60d` - Total precipitation (mm) in 60 days before repair
- `MeanTemp` - Average temperature during repair period
- `SnowOnGround` - Snow accumulation (cm) at time of repair

**Target Variable:**
- Binary classification: Will this pothole require re-repair within 6 months? (Yes/No)

## Geospatial Processing

The project leverages geospatial analysis to integrate data from multiple sources:

**Spatial Matching:**
- Used **GeoPandas** for handling spatial data (pothole locations, road networks, weather stations)
- Implemented **cKDTree** (SciPy) for efficient nearest-neighbor searches to match:
  - Potholes → Road segments (for road condition metrics)
  - Potholes → Weather stations (for local weather data)
- Performed **spatial joins** using Shapely geometry operations

**Data Integration Pipeline:**
1. Convert pothole coordinates to GeoPandas Point geometries
2. Use cKDTree to find nearest road segment for each pothole
3. Extract road condition features (PCI, IRI, age, surface type)
4. Match to nearest weather station for environmental features
5. Aggregate temporal weather data (30-day and 60-day windows)
6. Merge all features into final training dataset

This approach handles the spatial complexity of matching ~700,00 point locations to road network segments and weather stations efficiently.

## Model Performance

**Algorithm:** Random Forest Classifier

**Results:**
```
Overall Accuracy: 62%

Class 0 (No Recurrence):
- Precision: 0.78
- Recall: 0.68
- F1-Score: 0.73

Class 1 (Recurrence):
- Precision: 0.30
- Recall: 0.42
- F1-Score: 0.35
```

**Confusion Matrix:**
```
                Predicted No    Predicted Yes
Actual No          71,301         33,377
Actual Yes         19,654         14,182
```

## Key Insights

- The model correctly identifies **68% of non-recurring potholes**, reducing unnecessary deep repairs
- **42% recall on recurring potholes** helps flag high-risk repairs for enhanced maintenance
- Class imbalance (75% non-recurrence vs 25% recurrence) presents opportunities for improvement
- Freeze-thaw cycles and pavement condition indices are strong predictors of recurrence

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Workflow
The analysis is organized into sequential Jupyter notebooks:

1. **`data_cleaning.ipynb`** - Load and clean raw pothole, road, and weather data
2. **`feature_engineering.ipynb`** - Create spatial features using GeoPandas and weather aggregations
3. **`target_setting.ipynb`** - Define target variable (6-month recurrence)
4. **`pothole_prediction_model.ipynb`** - Train Random Forest model and evaluate performance

Run notebooks in order to reproduce the full analysis pipeline.

## Project Structure
```
pothole-prediction/
│
├── data_cleaning.ipynb              # Data preprocessing and cleaning
├── feature_engineering.ipynb        # Geospatial feature creation
├── target_setting.ipynb             # Target variable definition
├── pothole_prediction_model.ipynb   # Model training and evaluation
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## Future Improvements

- **Address class imbalance** - Implement SMOTE, class weighting, or cost-sensitive learning
- **Feature engineering** - Add traffic volume, road type, neighborhood socioeconomic factors
- **Model tuning** - Hyperparameter optimization via GridSearchCV, try XGBoost or LightGBM
- **Ensemble methods** - Combine multiple models for improved predictions
- **Deployment** - Create REST API for real-time predictions on new repairs
- **Feature importance analysis** - Visualize which factors most influence recurrence

## Real-World Applications

This model can be integrated into municipal maintenance management systems to:
- Flag high-risk repairs for follow-up inspections within 3-4 months
- Optimize repair crew scheduling and resource allocation
- Support capital planning for road resurfacing priorities
- Reduce long-term maintenance costs through preventive intervention
- Provide data-driven justification for infrastructure investment

## Data Sources & Acknowledgments

- [Montreal Open Data Portal](https://donnees.montreal.ca/) - Pothole repairs and road network data
- [Environment and Climate Change Canada](https://climate.weather.gc.ca/) - Historical weather data
- City of Montreal - Road condition assessment metrics

---

**Developed as part of Data Science program at Concordia University**

*This project demonstrates end-to-end machine learning workflow including data collection, geospatial processing, feature engineering, model training, and evaluation.*
