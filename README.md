# B2S Mismatch and Primary School Closure Risk in Korea (2020–2024)
This project investigates whether Births-to-Seat (B2S) mismatch can predict the risk of primary school closure/merger in Korea, and whether high-risk areas are disproportionately concentrated in non-urban regions.

##  Project Structure
b2s_project/
├─ analysis.py
├─ korea_primary_dataset_2020.2024.csv
└─ outputs/

## 📊 Dataset Description

| Column | Description |
| region_code | Region identifier |
| region_name | Korean region name |
| region_en | English region name |
| region_type | CapitalArea / MetroCity / Province |
| region_type_flag | 1 = urban, 0 = non-urban |
| year | Year |
| births | Birth count |
| grade1_students | Grade 1 enrollment |
| elem_school | Number of primary schools |
| school_change | Yearly change in school count |
| closed_elem_year | Closure year |
| closed_elem_year_2020_2024 | Closure indicator |
| births_per_school | births / elem_school |
| grade1_per_school | grade1_students / elem_school |
| b2s_ratio | grade1_students / births |
| b2s_gap | B2S gap indicator |
| risk_index | Proposal risk index |
| estimated_flag | Estimation flag |

##  Data Processing Pipeline
- Clean numeric formatting and convert types  
- Compute core indicators:  
  - births_per_school  
  - grade1_per_school  
  - b2s_ratio  
  - b2s_z (standardized B2S)  
  - b2s_low_flag (bottom 20%)  
- Create next-year school decrease target variable  
- Add interaction term for fairness testing  

##  Analysis Components

### Analysis 1 — Structural Differences
Latest-year B2S comparison across regions.  
Output: `structural_b2s_2024_by_region.png`

### Analysis 2 — Threshold / Non-Linearity
Quantile grouping (Low/Middle/High) and closure rate comparison.  
Output: `threshold_b2s_quantile3_closure_rate.png`

### Analysis 3 — Fairness Across Region Types
Urban vs Non-urban × Low-B2S interaction and logistic regression:  
`y ~ b2s_z + region_type_flag + interaction`  
Outputs:  
`fairness_low_b2s_by_region_type.png`  
`fairness_b2s_curve_by_region_type.png`

### Analysis 4 — Predicting Next-Year Closure Risk
Predicts 2025 closure probability for each region.  
Output: `prediction_next_year_by_region.png`

### Install dependencies:
pip install numpy pandas matplotlib statsmodels

### Run:
python analysis.py

### Outputs:
- Console results  
- Figures saved in `outputs/`
