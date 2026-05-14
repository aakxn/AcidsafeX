import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score, classification_report

import joblib

# -------------------------
# LOAD DATASET
# -------------------------

data = pd.read_csv("../industrial_acid_safety_dataset_12000.csv")

# -------------------------
# RENAME COLUMNS
# -------------------------

data.columns = [

    'acid_type',

    'quantity',

    'storage_temperature',

    'container_material',

    'exposure_time',

    'ventilation_quality',

    'safety_equipment',

    'storage_area',

    'previous_incident',

    'risk_level'
]

# -------------------------
# FEATURES
# -------------------------

features = [

    'acid_type',

    'quantity',

    'storage_temperature',

    'container_material',

    'exposure_time',

    'ventilation_quality',

    'safety_equipment',

    'storage_area',

    'previous_incident'
]

# -------------------------
# INPUT / OUTPUT
# -------------------------

X = data[features]

y = data['risk_level']

# -------------------------
# CATEGORICAL COLUMNS
# -------------------------

categorical_cols = [

    'acid_type',

    'container_material',

    'ventilation_quality',

    'safety_equipment',

    'storage_area',

    'previous_incident'
]

# -------------------------
# PREPROCESSOR
# -------------------------

preprocessor = ColumnTransformer(

    transformers=[

        (

            'cat',

            OneHotEncoder(handle_unknown='ignore'),

            categorical_cols

        )

    ],

    remainder='passthrough'
)

# -------------------------
# MODEL PIPELINE
# -------------------------

model = Pipeline([

    ('preprocessor', preprocessor),

    ('classifier', RandomForestClassifier(

        n_estimators=200,

        random_state=42

    ))
])

# -------------------------
# TRAIN TEST SPLIT
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42
)

# -------------------------
# TRAIN MODEL
# -------------------------

print("\nTraining model...\n")

model.fit(X_train, y_train)

# -------------------------
# PREDICTIONS
# -------------------------

predictions = model.predict(X_test)

# -------------------------
# EVALUATION
# -------------------------

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")

print("MODEL TRAINING COMPLETED")

print("==============================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(classification_report(y_test, predictions))

# -------------------------
# SAVE MODEL
# -------------------------

joblib.dump(model, "acid_risk_model.pkl")

print("\nModel saved successfully as acid_risk_model.pkl")