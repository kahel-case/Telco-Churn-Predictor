"""Telco Customer Churn Predictor - Simple Desktop App (Tkinter).
Uses YOUR trained model: rf_telco_tuned_model.pkl
Run: python churn_gui.py
"""
import json
import os
import warnings
import tkinter as tk
from tkinter import ttk, messagebox

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(HERE, "rf_telco_tuned_model.pkl")
ENCODER_FILE = os.path.join(HERE, "telco_encoder.pkl")
ORDER_FILE = os.path.join(HERE, "feature_order.json")
CATS_FILE = os.path.join(HERE, "cat_columns.json")
METRICS_FILE = os.path.join(HERE, "metrics_paper.json")

# ---------- load artifacts ----------
try:
    import joblib
    import pandas as pd
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}\nRun: pip install -r requirements.txt")

warnings.filterwarnings("ignore")  # silence sklearn InconsistentVersionWarning (1.6.1 vs 1.9.1)

try:
    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    with open(ORDER_FILE) as f:
        FEATURE_ORDER = json.load(f)
    with open(CATS_FILE) as f:
        CAT_COLS = json.load(f)
    with open(METRICS_FILE) as f:
        METRICS = json.load(f)
except Exception as e:
    raise SystemExit(f"Model files not found in {HERE}.\nDetails: {e}")


def predict_row(row: dict):
    """Encode + predict. Returns (pred_int, proba_churn)."""
    X = pd.DataFrame([row])[FEATURE_ORDER]
    X[CAT_COLS] = encoder.transform(X[CAT_COLS])
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    return pred, proba


def risk_band(pct: float) -> str:
    if pct >= 70:
        return "HIGH risk"
    if pct >= 40:
        return "MEDIUM risk"
    return "LOW risk"


def explain(row: dict) -> str:
    reasons = []
    if row["Contract"] == "Month-to-month":
        reasons.append("month-to-month contract (no lock-in)")
    if row["InternetService"] == "Fiber optic" and float(row["MonthlyCharges"]) >= 70:
        reasons.append("fiber optic + high monthly bill")
    if int(row["tenure"]) <= 6:
        reasons.append(f"short tenure ({row['tenure']} mo)")
    if row["TechSupport"] == "No" and row["InternetService"] != "No":
        reasons.append("no tech support despite having internet")
    if row["OnlineSecurity"] == "No" and row["InternetService"] != "No":
        reasons.append("no online security add-on")
    if row["PaymentMethod"] == "Electronic check":
        reasons.append("electronic-check payment (highest-churn segment)")
    if not reasons:
        reasons.append("long tenure + contract + add-on services (protective factors)")
    return "; ".join(reasons)


# ---------- GUI ----------
root = tk.Tk()
root.title("Telco Customer Churn Predictor")
root.geometry("900x760")

ttk.Label(root, text="Telco Customer Churn Predictor", font=("Segoe UI", 14, "bold")).pack(pady=6)
ttk.Label(root, text="Enter all 19 features, press Predict. Model: rf_telco_tuned_model.pkl (tuned Random Forest).").pack()

# input frame
inp = ttk.LabelFrame(root, text="1. Customer input - all necessary features (19)")
inp.pack(fill="x", padx=10, pady=6)

OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

vars_ = {}
row_i = 0
# row 0: gender, SeniorCitizen, Partner, Dependents
# Use grid: 4 columns of label+widget
fields_left = [
    ("gender", "combo", "Female"),
    ("SeniorCitizen", "senior", "0"),
    ("Partner", "combo", "Yes"),
    ("Dependents", "combo", "No"),
    ("tenure (months 0-72)", "tenure", "12"),
    ("PhoneService", "combo", "Yes"),
    ("MultipleLines", "combo", "No"),
    ("InternetService", "combo", "Fiber optic"),
    ("OnlineSecurity", "combo", "No"),
    ("OnlineBackup", "combo", "No"),
    ("DeviceProtection", "combo", "No"),
    ("TechSupport", "combo", "No"),
    ("StreamingTV", "combo", "No"),
    ("StreamingMovies", "combo", "No"),
    ("Contract", "combo", "Month-to-month"),
    ("PaperlessBilling", "combo", "Yes"),
    ("PaymentMethod", "combo", "Electronic check"),
    ("MonthlyCharges ($)", "monthly", "70.0"),
    ("TotalCharges ($)", "total", "800.0"),
]

for idx, (label, kind, default) in enumerate(fields_left):
    r, c = divmod(idx, 3)
    f = ttk.Frame(inp)
    f.grid(row=r, column=c, padx=8, pady=4, sticky="w")
    ttk.Label(f, text=label).pack(anchor="w")
    if kind == "combo":
        key = label  # label == feature name for combos
        v = tk.StringVar(value=default)
        cb = ttk.Combobox(f, textvariable=v, values=OPTIONS[key], state="readonly", width=24)
        cb.pack()
        vars_[key] = v
    elif kind == "senior":
        v = tk.StringVar(value=default)
        ttk.Combobox(f, textvariable=v, values=["0", "1"], state="readonly", width=24).pack()
        vars_["SeniorCitizen"] = v
    elif kind == "tenure":
        v = tk.StringVar(value=default)
        ttk.Spinbox(f, from_=0, to=72, textvariable=v, width=26).pack()
        vars_["tenure"] = v
    else:
        v = tk.StringVar(value=default)
        ttk.Entry(f, textvariable=v, width=27).pack()
        if kind == "monthly":
            vars_["MonthlyCharges"] = v
        else:
            vars_["TotalCharges"] = v

# ---------- output ----------
out = ttk.LabelFrame(root, text="2. Output - prediction")
out.pack(fill="x", padx=10, pady=6)
result_var = tk.StringVar(value="Press Predict Churn.")
ttk.Label(out, textvariable=result_var, font=("Segoe UI", 11, "bold"), wraplength=840).pack(padx=8, pady=4)
risk_var = tk.StringVar(value="")
ttk.Label(out, textvariable=risk_var, wraplength=840).pack(padx=8)
explain_var = tk.StringVar(value="")
ttk.Label(out, textvariable=explain_var, wraplength=840).pack(padx=8, pady=4)


def on_predict():
    try:
        try:
            monthly = float(vars_["MonthlyCharges"].get())
            total_raw = vars_["TotalCharges"].get().strip()
            total = float(total_raw) if total_raw != "" else 0.0  # blank -> 0 per paper 2.2.1
            tenure = int(float(vars_["tenure"].get()))
            senior = int(vars_["SeniorCitizen"].get())
        except ValueError:
            messagebox.showerror("Invalid input", "tenure / SeniorCitizen / MonthlyCharges / TotalCharges must be numbers.")
            return
        row = {
            "gender": vars_["gender"].get(),
            "SeniorCitizen": senior,
            "Partner": vars_["Partner"].get(),
            "Dependents": vars_["Dependents"].get(),
            "tenure": tenure,
            "PhoneService": vars_["PhoneService"].get(),
            "MultipleLines": vars_["MultipleLines"].get(),
            "InternetService": vars_["InternetService"].get(),
            "OnlineSecurity": vars_["OnlineSecurity"].get(),
            "OnlineBackup": vars_["OnlineBackup"].get(),
            "DeviceProtection": vars_["DeviceProtection"].get(),
            "TechSupport": vars_["TechSupport"].get(),
            "StreamingTV": vars_["StreamingTV"].get(),
            "StreamingMovies": vars_["StreamingMovies"].get(),
            "Contract": vars_["Contract"].get(),
            "PaperlessBilling": vars_["PaperlessBilling"].get(),
            "PaymentMethod": vars_["PaymentMethod"].get(),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        pred, proba = predict_row(row)
        pct = proba * 100
        label = "CHURN (Yes)" if pred == 1 else "NO CHURN (No)"
        result_var.set(f"Prediction: {label}  -  churn probability {pct:.1f}%  (stay {(100 - pct):.1f}%)")
        risk_var.set(f"Risk band: {risk_band(pct)}")
        prefix = "Model leans CHURN mainly because of: " if pred == 1 else "Model leans STAY mainly because of: "
        explain_var.set("How to interpret: " + prefix + explain(row) + ".")
    except Exception as e:
        messagebox.showerror("Prediction failed", str(e))


ttk.Button(root, text="Predict Churn", command=on_predict).pack(pady=4, ipadx=20, ipady=4)

# ---------- metrics ----------
met = ttk.LabelFrame(root, text="3. Metrics & percentages (paper Table 4, test set)")
met.pack(fill="both", expand=True, padx=10, pady=6)
cols = ["Model", "Accuracy %", "Precision %", "Recall %", "F1 %", "ROC-AUC"]
tree = ttk.Treeview(met, columns=cols, show="headings", height=2)
for col in cols:
    tree.heading(col, text=col)
    tree.column(col, width=130 if col != "Model" else 300)
tree.pack(fill="x", padx=8, pady=4)
for m in METRICS:
    tree.insert("", "end", values=(
        m["Model"],
        f"{m['Accuracy'] * 100:.1f} %",
        f"{m['Precision (Churn)'] * 100:.1f} %",
        f"{m['Recall (Churn)'] * 100:.1f} %",
        f"{m['F1-score (Churn)'] * 100:.1f} %",
        f"{m['ROC-AUC']:.3f}",
    ))
ttk.Label(met, text="Recall/F1 prioritized (missing a churner costs more). Tuning: n_estimators, max_depth, "
          "min_samples_split/leaf, max_features, class_weight. SMOTENC on train only.",
          wraplength=840).pack(padx=8, pady=4)

root.mainloop()
