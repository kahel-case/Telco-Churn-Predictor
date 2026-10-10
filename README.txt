Telco Customer Churn Predictor (Desktop, Tkinter)
===============================================
Simple app using the trained model: rf_telco_tuned_model.pkl
(Random Forest, SMOTENC + RandomizedSearchCV tuning, from CST9L Final Project).

Run:
  python churn_gui.py

Files in this folder:
  churn_gui.py               - Tkinter app (19 inputs + 1 Predict button + metrics %)
  rf_telco_tuned_model.pkl   - the tuned model (500 trees, max_depth 20, log2, balanced)
  telco_encoder.pkl          - OrdinalEncoder for 15 categorical columns (reused, verified compatible)
  feature_order.json         - 19 feature names in model order
  cat_columns.json           - 15 categorical column names
  metrics_paper.json         - paper Table 4 metrics (shown as percentages in app)
  requirements.txt           - pandas, scikit-learn, joblib

Notes:
- If you see InconsistentVersionWarning (model pickled with sklearn 1.6.1,
  you run 1.9.1), it still predicts correctly. For exact match: pip install -r requirements.txt
- TotalCharges blank -> treated as 0 (tenure 0 customers, per paper section 2.2.1).
- Probability = share of trees voting churn.
