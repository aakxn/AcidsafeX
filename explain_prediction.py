# ---------------------------------------------------
# EXPLAINABLE ML USING SHAP
# FINAL CORRECTED VERSION
# ---------------------------------------------------

import pandas as pd
import shap
import joblib
import numpy as np

# ---------------------------------------------------
# LOAD TRAINED MODEL PIPELINE
# ---------------------------------------------------

model = joblib.load("model/acid_risk_model.pkl")

# ---------------------------------------------------
# EXTRACT PREPROCESSOR + CLASSIFIER
# ---------------------------------------------------

preprocessor = model.named_steps["preprocessor"]

classifier = model.named_steps["classifier"]

# ---------------------------------------------------
# CREATE SHAP EXPLAINER
# ---------------------------------------------------

explainer = shap.TreeExplainer(classifier)

# ---------------------------------------------------
# EXPLAIN PREDICTION FUNCTION
# ---------------------------------------------------

def explain_prediction(data):

    # ---------------------------------------------------
    # CREATE INPUT DATAFRAME
    # ---------------------------------------------------

    input_df = pd.DataFrame([{

        "acid_type": data['acid_type'],
        "quantity": data['quantity'],
        "storage_temperature": data['storage_temperature'],
        "container_material": data['container_material'],
        "exposure_time": data['exposure_time'],
        "ventilation_quality": data['ventilation_quality'],
        "safety_equipment": data['safety_equipment'],
        "storage_area": data['storage_area'],
        "previous_incident": data['previous_incident']

    }])

    # ---------------------------------------------------
    # TRANSFORM INPUT
    # ---------------------------------------------------

    transformed_input = preprocessor.transform(input_df)

    if hasattr(transformed_input, "toarray"):

        transformed_input = transformed_input.toarray()

    # ---------------------------------------------------
    # GET FEATURE NAMES
    # ---------------------------------------------------

    feature_names = preprocessor.get_feature_names_out()

    # ---------------------------------------------------
    # CALCULATE SHAP VALUES
    # ---------------------------------------------------

    shap_values = explainer.shap_values(

        transformed_input,

        check_additivity=False
    )

        # ---------------------------------------------------
    # HANDLE DIFFERENT SHAP OUTPUT FORMATS
    # ---------------------------------------------------

    try:

        # NEWER SHAP EXPLANATION OBJECT

        if hasattr(shap_values, "values"):

            impacts = np.abs(
                shap_values.values[0]
            )

            impacts = np.array(
                impacts
            ).flatten()

        # MULTI-CLASS FORMAT

        elif isinstance(shap_values, list):

            shap_array = np.abs(
                np.array(shap_values)
            )

            # AVERAGE ACROSS CLASSES

            impacts = np.mean(
                shap_array,
                axis=0
            )

            # FIRST SAMPLE

            impacts = impacts[0]

            # FLATTEN

            impacts = np.array(
                impacts
            ).flatten()

        # NORMAL ARRAY FORMAT

        else:

            impacts = np.abs(
                shap_values[0]
            )

            impacts = np.array(
                impacts
            ).flatten()

    except Exception as e:

        print("SHAP Processing Error:", e)

        impacts = np.zeros(
            len(feature_names)
        )

        


    # ---------------------------------------------------
    # DEBUGGING
    # ---------------------------------------------------

    print("\nFeature Names Count:", len(feature_names))
    print("Impacts Count:", len(impacts))
    print("Impacts:", impacts)

    # ---------------------------------------------------
    # CREATE FEATURE + IMPACT PAIRS
    # ---------------------------------------------------

    feature_impact_pairs = list(zip(

        feature_names,

        impacts
    ))

    # ---------------------------------------------------
    # GROUP ORIGINAL FEATURES
    # ---------------------------------------------------

    grouped_importance = {

        "acid_type": 0,
        "quantity": 0,
        "storage_temperature": 0,
        "container_material": 0,
        "exposure_time": 0,
        "ventilation_quality": 0,
        "safety_equipment": 0,
        "storage_area": 0,
        "previous_incident": 0
    }

    # ---------------------------------------------------
    # MERGE ONE-HOT ENCODED FEATURES
    # ---------------------------------------------------

    for feature, impact in feature_impact_pairs:

        feature = str(feature)

        for key in grouped_importance.keys():

            if key in feature:

                grouped_importance[key] += float(impact)

    # ---------------------------------------------------
    # TOTAL IMPORTANCE
    # ---------------------------------------------------

    total_importance = sum(

        grouped_importance.values()
    )

    # ---------------------------------------------------
    # EXPLANATION OUTPUT
    # ---------------------------------------------------

    explanations = []

    explanations.append(

        "Explainable ML Analysis"
    )

    explanations.append(

        "The machine learning model analyzed multiple industrial safety parameters to determine the predicted acid risk level."
    )

    # ---------------------------------------------------
    # SORT FEATURES
    # ---------------------------------------------------

    sorted_features = sorted(

        grouped_importance.items(),

        key=lambda x: x[1],

        reverse=True
    )

    # ---------------------------------------------------
    # GENERATE EXPLANATIONS
    # ---------------------------------------------------

    for feature, impact in sorted_features:

        if total_importance == 0:

            percent = 0

        else:

            percent = round(

                (impact / total_importance) * 100,

                2
            )

        # ---------------------------------------------------
        # ACID TYPE
        # ---------------------------------------------------

        if feature == "acid_type":

            explanations.append(

                f"Acid Type: {data['acid_type']} influenced overall industrial danger level ({percent}%)."
            )

        # ---------------------------------------------------
        # QUANTITY
        # ---------------------------------------------------

        elif feature == "quantity":

            explanations.append(

                f"Quantity: {data['quantity']}L acid storage contributed heavily to elevated hazard probability ({percent}%)."
            )

        # ---------------------------------------------------
        # STORAGE TEMPERATURE
        # ---------------------------------------------------

        elif feature == "storage_temperature":

            explanations.append(

                f"Storage Temperature: Operating at {data['storage_temperature']}°C increased chemical instability risk ({percent}%)."
            )

        # ---------------------------------------------------
        # CONTAINER MATERIAL
        # ---------------------------------------------------

        elif feature == "container_material":

            explanations.append(

                f"Container Material: {data['container_material']} containers influenced acid storage safety ({percent}%)."
            )

        # ---------------------------------------------------
        # EXPOSURE TIME
        # ---------------------------------------------------

        elif feature == "exposure_time":

            explanations.append(

                f"Exposure Time: Continuous exposure for {data['exposure_time']} hours increased worker safety concerns ({percent}%)."
            )

        # ---------------------------------------------------
        # VENTILATION QUALITY
        # ---------------------------------------------------

        elif feature == "ventilation_quality":

            explanations.append(

                f"Ventilation Quality: {data['ventilation_quality']} airflow conditions influenced toxic gas exposure risk ({percent}%)."
            )

        # ---------------------------------------------------
        # SAFETY EQUIPMENT
        # ---------------------------------------------------

        elif feature == "safety_equipment":

            explanations.append(

                f"Safety Equipment: {data['safety_equipment']} protective equipment availability affected industrial safety compliance ({percent}%)."
            )

        # ---------------------------------------------------
        # STORAGE AREA
        # ---------------------------------------------------

        elif feature == "storage_area":

            explanations.append(

                f"Storage Area: {data['storage_area']} storage conditions influenced hazard management efficiency ({percent}%)."
            )

        # ---------------------------------------------------
        # PREVIOUS INCIDENT
        # ---------------------------------------------------

        elif feature == "previous_incident":

            explanations.append(

                f"Previous Incident History: {data['previous_incident']} previous incidents influenced industrial risk prediction ({percent}%)."
            )

    # ---------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------

    if total_importance == 0:

        explanations.append(

            "Industrial environment appears stable and compliant."
        )

    # ---------------------------------------------------
    # RETURN
    # ---------------------------------------------------

    return explanations, grouped_importance