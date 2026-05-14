
import sqlite3
import pandas as pd

# -------------------------
# CONNECT DATABASE
# -------------------------

conn = sqlite3.connect('instance/database.db')

# -------------------------
# EXPORT DATASET
# -------------------------

query = """

SELECT

    acid_record.id,

    user.username,

    user.email,

    user.role,

    acid_record.acid_type,

    acid_record.quantity,

    acid_record.storage_temperature,

    acid_record.container_material,

    acid_record.exposure_time,

    acid_record.ventilation_quality,

    acid_record.safety_equipment,

    acid_record.storage_area,

    acid_record.previous_incident,

    acid_record.risk_level,

    acid_record.compliance_score,

    acid_record.date

FROM acid_record

LEFT JOIN user

ON acid_record.user_id = user.id

"""

# -------------------------
# LOAD INTO DATAFRAME
# -------------------------

df = pd.read_sql_query(
    query,
    conn
)

# -------------------------
# SAVE CSV
# -------------------------

df.to_csv(

    "industrial_acid_safety_dataset_12000.csv",

    index=False

)

# -------------------------
# CLOSE DATABASE
# -------------------------

conn.close()

# -------------------------
# SUCCESS MESSAGE
# -------------------------

print("\n====================================")
print("DATASET EXPORT COMPLETED SUCCESSFULLY")
print("====================================\n")

print("File Saved:")
print("industrial_acid_safety_dataset_12000.csv\n")

print("Total Records Exported:")
print(len(df))

print("\nColumns Exported:\n")
print(df.columns.tolist())

