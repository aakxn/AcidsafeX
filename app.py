
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import joblib
import os
import json

# Explainable AI
from model.explain_prediction import explain_prediction

app = Flask(__name__)
app.secret_key = "acidsafex_secret_key"

# -------------------------
# BASE DIRECTORY
# -------------------------

basedir = os.path.abspath(os.path.dirname(__file__))

# -------------------------
# LOAD ML MODEL
# -------------------------

model = joblib.load("model/acid_risk_model.pkl")

# -------------------------
# DATABASE CONFIG
# -------------------------

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(
    basedir,
    'instance',
    'database.db'
)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# -------------------------
# USER TABLE
# -------------------------

class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(100))

    email = db.Column(db.String(100), unique=True)

    password = db.Column(db.String(200))

    role = db.Column(db.String(50), default="Worker")

    # RELATIONSHIP
    records = db.relationship(
        'AcidRecord',
        backref='user',
        lazy=True
    )


# -------------------------
# ACID RECORD TABLE
# -------------------------

class AcidRecord(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    acid_type = db.Column(db.String(100))

    quantity = db.Column(db.Float)

    storage_temperature = db.Column(db.Float)

    container_material = db.Column(db.String(100))

    exposure_time = db.Column(db.Float)

    ventilation_quality = db.Column(db.String(100))

    safety_equipment = db.Column(db.String(200))

    storage_area = db.Column(db.String(100))

    previous_incident = db.Column(db.String(100))

    risk_level = db.Column(db.String(20))

    compliance_score = db.Column(db.Float)

    explanation = db.Column(db.Text)

    precautions = db.Column(db.Text)

    feature_importance = db.Column(db.Text)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id')
    )

    date = db.Column(
        db.DateTime,
        default=datetime.datetime.utcnow
    )



 

# -------------------------
# ML PREDICTION
# -------------------------

def predict_risk_ml(data):

    import pandas as pd

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

    prediction = model.predict(input_df)[0]

    prediction = str(prediction).upper().strip()

    if prediction == "CRITICAL RISK":

        return "Critical"

    elif prediction == "HIGH RISK":

        return "High"

    elif prediction == "MEDIUM RISK":

        return "Medium"

    else:

        return "Low"


# -------------------------
# LOGIN
# -------------------------

@app.route('/', methods=['GET', 'POST'])
def login():

    if request.method == "POST":

        email = request.form['email']

        password = request.form['password']

        user = User.query.filter_by(
            email=email
        ).first()

        if user and check_password_hash(
            user.password,
            password
        ):

            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role

            return redirect(
                url_for('dashboard')
            )

        else:

            return render_template(
                "login.html",
                error="Invalid email or password"
            )

    return render_template("login.html")


# -------------------------
# SIGNUP
# -------------------------

@app.route('/signup', methods=['GET', 'POST'])
def signup():

    if request.method == 'POST':

        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:

            return render_template(
                'signup.html',
                error="Email already registered"
            )

        hashed_password = generate_password_hash(password)

        new_user = User(

            username=username,
            email=email,
            password=hashed_password,
            role=role

        )

        db.session.add(new_user)
        db.session.commit()

        return redirect('/')

    return render_template('signup.html')


# -------------------------
# DASHBOARD
# -------------------------

@app.route('/dashboard')
def dashboard():

    if 'user_id' not in session:

        return redirect(url_for('login'))

    user = User.query.get(
        session['user_id']
    )

    # -------------------------
    # COMMON COUNTS
    # -------------------------

    critical_count = AcidRecord.query.filter_by(
        risk_level="Critical"
    ).count()

    high_count = AcidRecord.query.filter_by(
        risk_level="High"
    ).count()

    medium_count = AcidRecord.query.filter_by(
        risk_level="Medium"
    ).count()

    low_count = AcidRecord.query.filter_by(
        risk_level="Low"
    ).count()

    # -------------------------
    # ADMIN DASHBOARD
    # -------------------------

    if user.role == "Admin":

        records = AcidRecord.query.order_by(
            AcidRecord.date.desc()
        ).all()

        total_users = User.query.count()

        total_records = AcidRecord.query.count()

        avg_compliance = db.session.query(
            db.func.avg(
                AcidRecord.compliance_score
            )
        ).scalar()

        safety_officers = User.query.filter_by(
            role="Safety Officer"
        ).count()

        workers = User.query.filter_by(
            role="Worker"
        ).count()

        total_assessments = total_records

        return render_template(

            "dashboard.html",

            user=user,

            records=records,

            total_users=total_users,

            total_records=total_records,

            critical_count=critical_count,

            high_count=high_count,

            medium_count=medium_count,

            low_count=low_count,

            avg_compliance=round(
                avg_compliance or 0,
                2
            ),

            safety_officers=safety_officers,

            workers=workers,

            total_assessments=total_assessments
        )

    # -------------------------
    # SAFETY OFFICER DASHBOARD
    # -------------------------

    elif user.role == "Safety Officer":

        records = AcidRecord.query.order_by(
            AcidRecord.date.desc()
        ).all()

        high_risk_today = AcidRecord.query.filter(
            AcidRecord.risk_level.in_(
                ["Critical", "High"]
            )
        ).count()

        total_staff = User.query.filter_by(
            role="Worker"
        ).count()

        total_assessments = AcidRecord.query.count()

        return render_template(

            "dashboard.html",

            user=user,

            records=records,

            high_risk_today=high_risk_today,

            total_staff=total_staff,

            total_assessments=total_assessments,

            critical_count=critical_count,

            high_count=high_count,

            medium_count=medium_count,

            low_count=low_count
        )

    # -------------------------
    # WORKER DASHBOARD
    # -------------------------

    else:

        records = AcidRecord.query.filter_by(
            user_id=user.id
        ).order_by(
            AcidRecord.date.desc()
        ).all()

        total_assessments = len(records)

        # ONLY WORKER'S OWN COUNTS

        critical_count = AcidRecord.query.filter_by(
            user_id=user.id,
            risk_level="Critical"
        ).count()

        high_count = AcidRecord.query.filter_by(
            user_id=user.id,
            risk_level="High"
        ).count()

        medium_count = AcidRecord.query.filter_by(
            user_id=user.id,
            risk_level="Medium"
        ).count()

        low_count = AcidRecord.query.filter_by(
            user_id=user.id,
            risk_level="Low"
        ).count()

        return render_template(

            "dashboard.html",

            user=user,

            records=records,

            total_assessments=total_assessments,

            critical_count=critical_count,

            high_count=high_count,

            medium_count=medium_count,

            low_count=low_count
        )  
   

# -------------------------
# FORM PAGE
# -------------------------

@app.route('/form')
def form():

    if 'user_id' not in session:

        return redirect(url_for('login'))

    # WORKER ALSO ALLOWED NOW

    return render_template(
        "form.html"
    )



# -------------------------
# SUBMIT FORM
# -------------------------

@app.route('/submit', methods=['POST'])
def submit():

    if 'user_id' not in session:

        return redirect(url_for('login'))

    acid = request.form['acid_type']

    quantity = float(
        request.form['quantity']
    )

    storage_temperature = float(
        request.form['storage_temperature']
    )

    exposure = float(
        request.form['exposure_time']
    )

    container = request.form[
        'container_material'
    ]

    ventilation_quality = request.form[
        'ventilation_quality'
    ]

    safety = request.form[
        'safety_equipment'
    ]

    storage = request.form[
        'storage_area'
    ]

    incident = request.form[
        'previous_incident'
    ]

    data = {

        "acid_type": acid,
        "quantity": quantity,
        "storage_temperature": storage_temperature,
        "container_material": container,
        "exposure_time": exposure,
        "ventilation_quality": ventilation_quality,
        "safety_equipment": safety,
        "storage_area": storage,
        "previous_incident": incident
    }

    # -------------------------
    # ML RISK PREDICTION
    # -------------------------

    risk = predict_risk_ml(data)

    
        # -------------------------
    # EXPLAINABLE AI
    # -------------------------

    explanation = []

    feature_importance = {

        "acid_type": 10,

        "quantity": min(quantity / 10, 25),

        "storage_temperature": min(storage_temperature / 5, 25),

        "container_material": 8,

        "exposure_time": min(exposure / 8, 20),

        "ventilation_quality": 15 if "poor" in ventilation_quality.lower() else 6,

        "safety_equipment": 18 if "none" in safety.lower() else 8,

        "storage_area": 10,

        "previous_incident": 20 if incident.lower() != "none" else 5

    }

    # -------------------------
    # BASE EXPLANATIONS
    # ALWAYS SHOW ALL PARAMETERS
    # -------------------------

    explanation.append(
        f"Acid Type: {acid} influenced the overall industrial chemical hazard level."
    )

    explanation.append(
        f"Quantity: {quantity}L acid storage affected industrial hazard probability."
    )

    explanation.append(
        f"Storage Temperature: {storage_temperature}°C influenced chemical stability and thermal reaction risk."
    )

    explanation.append(
        f"Container Material: {container} containers impacted acid storage protection efficiency."
    )

    explanation.append(
        f"Exposure Time: {exposure} hours exposure affected worker safety conditions."
    )

    explanation.append(
        f"Ventilation Quality: {ventilation_quality} airflow conditions influenced toxic gas accumulation risk."
    )

    explanation.append(
        f"Safety Equipment: {safety} protective measures affected industrial compliance standards."
    )

    explanation.append(
        f"Storage Area: {storage} influenced hazardous chemical management efficiency."
    )

    explanation.append(
        f"Previous Incident History: {incident} affected industrial accident prediction analysis."
    )

    # -------------------------
    # ADVANCED DYNAMIC LOGIC
    # -------------------------

    if quantity >= 150:

        feature_importance["quantity"] += 10

        explanation.append(
            f"High Quantity Alert: {quantity}L storage volume significantly increased industrial hazard severity."
        )

    if storage_temperature >= 70:

        feature_importance["storage_temperature"] += 10

        explanation.append(
            f"Temperature Warning: {storage_temperature}°C storage conditions increased chemical instability risk."
        )

    if exposure >= 72:

        feature_importance["exposure_time"] += 8

        explanation.append(
            f"Extended Exposure Alert: {exposure} hours exposure duration elevated worker safety concerns."
        )

    if "poor" in ventilation_quality.lower():

        feature_importance["ventilation_quality"] += 12

        explanation.append(
            "Critical Ventilation Warning: Poor airflow conditions increased toxic gas exposure probability."
        )

    if incident.lower() != "none":

        feature_importance["previous_incident"] += 10

        explanation.append(
            f"Incident History Warning: Previous {incident} incidents increased future industrial accident probability."
        )

    if "none" in safety.lower() or "basic" in safety.lower():

        feature_importance["safety_equipment"] += 15

        explanation.append(
            f"Safety Compliance Warning: {safety} provided insufficient industrial protection measures."
        )

    # -------------------------
    # NORMALIZE PERCENTAGES
    # -------------------------

    total = sum(feature_importance.values())

    for key in feature_importance:

        feature_importance[key] = round(
            (feature_importance[key] / total) * 100,
            2
        )




    # -------------------------
    # COMPLIANCE SCORE
    # -------------------------

    if risk == "Critical":

        compliance_score = 10

    elif risk == "High":

        compliance_score = 35

    elif risk == "Medium":

        compliance_score = 65

    else:

        compliance_score = 90

    # -------------------------
    # PRECAUTIONS
    # -------------------------

    precautions = []

    if risk == "Critical":

        precautions.extend([

            "Immediate emergency response activation required.",
            "Evacuate nearby workers immediately.",
            "Use full-body PPE kits.",
            "Activate industrial ventilation systems."

        ])

    elif risk == "High":

        precautions.extend([

            "Improve ventilation quality immediately.",
            "Inspect containers for leakage.",
            "Keep emergency neutralization kits nearby."

        ])

    elif risk == "Medium":

        precautions.extend([

            "Maintain proper storage conditions.",
            "Ensure workers use gloves and eye protection."

        ])

    else:

        precautions.extend([

            "Continue standard acid handling procedures.",
            "Maintain periodic inspections."

        ])

    # -------------------------
    # SAVE RECORD
    # -------------------------

    new_record = AcidRecord(

        acid_type=acid,
        quantity=quantity,
        storage_temperature=storage_temperature,
        container_material=container,
        exposure_time=exposure,
        ventilation_quality=ventilation_quality,
        safety_equipment=safety,
        storage_area=storage,
        previous_incident=incident,
        risk_level=risk,
        compliance_score=compliance_score,
        user_id=session['user_id'],
        explanation=json.dumps(explanation), 
        precautions=json.dumps(precautions),
        feature_importance=json.dumps(feature_importance)

    )

    db.session.add(new_record)
    db.session.commit()

    return render_template(

        "result.html",

        risk=risk,

        compliance=compliance_score,

        explanation=explanation,

        precautions=precautions,

        feature_importance=feature_importance
    )


# -------------------------
# DELETE RECORD
# -------------------------

@app.route('/delete_record/<int:id>')
def delete_record(id):

    if 'user_id' not in session:

        return redirect(url_for('login'))

    user = User.query.get(
        session['user_id']
    )

    record = AcidRecord.query.get_or_404(id)

    if user.role == "Worker":

        return "Access Denied"

    db.session.delete(record)
    db.session.commit()

    return redirect(url_for('history'))


# -------------------------
# HISTORY
# -------------------------

@app.route('/history')
def history():

    if 'user_id' not in session:

        return redirect(url_for('login'))

    user = User.query.get(
        session['user_id']
    )

    if user.role == "Worker":

        records = AcidRecord.query.filter_by(
            user_id=user.id
        ).order_by(
            AcidRecord.date.desc()
        ).all()

    else:

        records = AcidRecord.query.order_by(
            AcidRecord.date.desc()
        ).all()

    return render_template(

        "history.html",

        records=records,

        user=user
    )



# -------------------------
# ACCOUNT
# -------------------------

@app.route('/account')
def account():

    if 'user_id' not in session:

        return redirect(url_for('login'))

    user = User.query.get(
        session['user_id']
    )

    return render_template(

        "account.html",

        user=user
    )

# =========================
# GENERATE REPORT
# =========================

@app.route('/generate_report/<int:id>')
def generate_report(id):

    if 'user_id' not in session:

        return redirect(url_for('login'))

    record = AcidRecord.query.get_or_404(id)

    user = User.query.get(record.user_id)

    explanation = json.loads(record.explanation)

    precautions = json.loads(record.precautions)

    feature_importance = json.loads(
        record.feature_importance
    )

    return render_template(

        "report.html",

        record=record,

        user=user,

        explanation=explanation,

        precautions=precautions,

        feature_importance=feature_importance
    )


# -------------------------
# LOGOUT
# -------------------------

@app.route('/logout')
def logout():

    session.clear()

    return redirect(url_for('login'))


# -------------------------
# RUN APP
# -------------------------

if __name__ == '__main__':

    with app.app_context():

        db.create_all()

    app.run(debug=True)

