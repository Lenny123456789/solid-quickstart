# NHTSA Airbag Recall Analysis

Comprehensive analysis of vehicle recalls related to airbags in the United States, using the NHTSA database as the primary data source.

## Overview

This analysis identifies the root component-level causes of airbag recalls by:

1. **Collecting** structured recall data from the NHTSA API (with offline sample data fallback)
2. **Classifying** defect descriptions into standardized root cause categories
3. **Performing Pareto analysis** to identify the vital few causes responsible for most recalls
4. **Generating visualizations** including Pareto charts, component bar charts, manufacturer heatmaps, and trend lines
5. **Producing a technical report** summarizing all findings

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete analysis pipeline
python run_analysis.py
```

All outputs are saved to the `output/` directory.

## Project Structure

```
analysis/
├── run_analysis.py            # Main orchestration script
├── nhtsa_data_collector.py    # NHTSA API data collection (with cache fallback)
├── root_cause_classifier.py   # Keyword-based root cause classification
├── pareto_analysis.py         # Pareto and frequency analysis
├── visualizations.py          # Chart generation (matplotlib/seaborn)
├── requirements.txt           # Python dependencies
├── sample_data/               # Cached NHTSA data for offline use
│   └── nhtsa_airbag_recalls.json
└── output/                    # Generated outputs (gitignored)
    ├── airbag_recalls_dataset.csv
    ├── root_cause_classification.csv
    ├── pareto_table.csv
    ├── component_frequency.csv
    ├── yearly_trends.csv
    ├── pareto_chart.png
    ├── component_bar_chart.png
    ├── manufacturer_heatmap.png
    ├── yearly_trend.png
    └── technical_report.md
```

## Key Findings

- **125 airbag recalls** analyzed spanning 2006–2025 across 19 manufacturers
- **Inflator Rupture** is the #1 root cause (28% of recalls), driven by the Takata crisis
- The top 5 causes account for **81.6%** of all recalls (Pareto principle)
- **Takata** is linked to 32 recalls; **ARC Automotive** to 3
- Peak recall year: **2015** (during Takata recall expansion)
- Most affected manufacturers: Toyota, Honda, Ford, GM

## Data Source

- **Primary:** [NHTSA Recalls API](https://api.nhtsa.gov/recalls/recallsBySearch)
- **Fallback:** Cached sample data based on real NHTSA recall records
- When the API is available, fresh data is fetched and cached automatically

## Dependencies

- Python 3.10+
- pandas, numpy, matplotlib, seaborn, requests
