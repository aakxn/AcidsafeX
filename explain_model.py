# -------------------------------------------------------
# EXPLAINABLE MACHINE LEARNING FOR ACID RISK PREDICTION
# OPTIMIZED VERSION
# -------------------------------------------------------

import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------

print("Loading dataset...\n")

data = pd.read_csv("../industrial_acid_safety_dataset_12000.csv")

print("Original Dataset Columns:\n")
print(data.columns.tolist())

# -------------------------------------------------------
# RENAME COLUMNS
# -------------------------------------------------------

data.columns = [

    "acid_type",
    "quantity",
    "storage_temperature",
    "container_material",
    "exposure_time",
    "ventilation_quality",
    "safety_equipment",
    "storage_area",
    "previous_incident",
    "risk_level"
]

print("\nRenamed Dataset Columns:\n")
print(data.columns.tolist())

# -------------------------------------------------------
# DEFINE FEATURES
# -------------------------------------------------------

features = [

    "acid_type",
    "quantity",
    "storage_temperature",
    "container_material",
    "exposure_time",
    "ventilation_quality",
    "safety_equipment",
    "storage_area",
    "previous_incident"
]

X = data[features]

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

print("\nLoading trained model...\n")

model = joblib.load("acid_risk_model.pkl")

# -------------------------------------------------------
# TRANSFORM DATA
# -------------------------------------------------------

print("Transforming dataset using pipeline...\n")

X_transformed = model.named_steps[
    "preprocessor"
].transform(X)

# -------------------------------------------------------
# CONVERT TO NUMERIC ARRAY
# -------------------------------------------------------

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

# -------------------------------------------------------
# TAKE SMALL SAMPLE
# -------------------------------------------------------

# VERY IMPORTANT FOR SPEED

sample_size = 200

if len(X_transformed) > sample_size:

    random_indices = np.random.choice(

        len(X_transformed),

        sample_size,

        replace=False
    )

    X_sample = X_transformed[random_indices]

else:

    X_sample = X_transformed

# -------------------------------------------------------
# GET CLASSIFIER
# -------------------------------------------------------

classifier = model.named_steps["classifier"]

# -------------------------------------------------------
# CREATE EXPLAINER
# -------------------------------------------------------

print("Creating SHAP explainer...\n")

explainer = shap.TreeExplainer(classifier)

# -------------------------------------------------------
# CALCULATE SHAP VALUES
# -------------------------------------------------------

print("Calculating SHAP values...\n")

shap_values = explainer.shap_values(
    X_sample,
    check_additivity=False
)

print("SHAP calculation completed successfully!\n")

# -------------------------------------------------------
# FEATURE NAMES
# -------------------------------------------------------

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

# -------------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------------

# -------------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------------

print("===================================")
print("FEATURE IMPORTANCE RANKING")
print("===================================\n")

# HANDLE MULTICLASS OUTPUT

if isinstance(shap_values, list):

    # Convert list to numpy array
    shap_array = np.array(shap_values)

    # Average across classes and samples
    importance = np.mean(
        np.abs(shap_array),
        axis=(0, 1)
    )

else:

    # For normal array output
    importance = np.mean(
        np.abs(shap_values),
        axis=0
    )

# FLATTEN ARRAY

importance = np.array(importance).flatten()

# CREATE FEATURE-IMPORTANCE PAIRS

feature_importance_pairs = list(zip(
    feature_names,
    importance
))

# SORT

sorted_importance = sorted(

    feature_importance_pairs,

    key=lambda x: float(x[1]),

    reverse=True
)

# PRINT TOP FEATURES

for feature, value in sorted_importance[:20]:

    readable_feature = feature

    readable_feature = readable_feature.replace(
        "cat__", ""
    )

    readable_feature = readable_feature.replace(
        "remainder__", ""
    )

    readable_feature = readable_feature.replace(
        "_",
        " "
    )

    print(f"{readable_feature} : {float(value):.4f}")



# -------------------------------------------------------
# SUMMARY PLOT
# -------------------------------------------------------

print("\nGenerating SHAP summary plot...\n")

plt.figure()

if isinstance(shap_values, list):

    shap.summary_plot(

        shap_values[0],

        X_sample,

        feature_names=feature_names,

        max_display=15,

        show=False
    )

else:

    shap.summary_plot(

        shap_values,

        X_sample,

        feature_names=feature_names,

        max_display=15,

        show=False
    )

plt.title("SHAP Summary Plot")

plt.tight_layout()

plt.show()

# -------------------------------------------------------
# BAR PLOT
# -------------------------------------------------------

print("\nGenerating feature importance bar plot...\n")

plt.figure()

if isinstance(shap_values, list):

    shap.summary_plot(

        shap_values[0],

        X_sample,

        feature_names=feature_names,

        plot_type="bar",

        max_display=15,

        show=False
    )

else:

    shap.summary_plot(

        shap_values,

        X_sample,

        feature_names=feature_names,

        plot_type="bar",

        max_display=15,

        show=False
    )

plt.title("SHAP Feature Importance")

plt.tight_layout()

plt.show()

# -------------------------------------------------------
# SAVE FEATURE IMPORTANCE CSV
# -------------------------------------------------------

importance_df = pd.DataFrame({

    "Feature": [

        f.replace("cat__", "").replace(
            "remainder__", ""
        )

        for f in feature_names
    ],

    "Importance": importance
})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False
)

importance_df.to_csv(

    "feature_importance.csv",

    index=False
)

print("\nFeature importance saved successfully.")

# -------------------------------------------------------
# COMPLETED
# -------------------------------------------------------

print("\n===================================")
print("EXPLAINABLE ML COMPLETED SUCCESSFULLY")
print("===================================")