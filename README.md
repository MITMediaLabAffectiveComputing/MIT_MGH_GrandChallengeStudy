# MGH_GCS_Study
MGH grand challenge study

# Setup
`pip install -r requirements.txt`

# Pipeline

## Creating features
for all features, run the respective .py file under feature_generation module. 
Run survey_label_individualization.py after running `survey_features.py`.
After extracting all features, run `combine_features.py`.

## Imputation

run `HDRS_imputation/imputation_survey.py`
the results will go to `factors/[measure]_imputation`

## Prediction

run `HDRS_prediction/ensemble_per_modality.py`
(You can change the modalities of interest in the `my_constants.py` file)
