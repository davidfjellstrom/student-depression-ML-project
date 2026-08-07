"""
Exporterar champion-modellen från 04_Supervised.ipynb till fil, så att app.py
kan ladda in den utan att träna om något.

Modellen är exakt densamma som i notebooken: samma features, samma stratifierade
80/20-split (random_state=42) och samma pipeline. Logistisk regression är
deterministisk, så resultatet blir identiskt varje körning - skriptet skriver ut
test-AUC på slutet så att du kan verifiera mot notebookens 0.8749.

Kör med:  ./venv/bin/python export_model.py
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Samma features som i 04_Supervised.ipynb - suicidal thoughts och City exkluderade.
FEATURE_COLS = [
    "Gender", "Age", "Academic Pressure", "CGPA", "Study Satisfaction",
    "Sleep Duration", "Dietary Habits", "Work/Study Hours", "Financial Stress",
    "Family History of Mental Illness",
]

df = pd.read_csv("student_depression_clean.csv")
X = df[FEATURE_COLS]
y = df["Depression"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

champion = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
champion.fit(X_train, y_train)

test_auc = roc_auc_score(y_test, champion.predict_proba(X_test)[:, 1])
print(f"Test-AUC: {test_auc:.4f}  (ska vara 0.8749 enligt 04_Supervised.ipynb)")

# Vi sparar även feature-ordningen - appen måste skicka in kolumnerna i samma
# ordning som modellen tränades på, annars blir prediktionerna tyst felaktiga.
joblib.dump({"pipeline": champion, "features": FEATURE_COLS}, "champion_model.joblib")
print("Sparad som champion_model.joblib")
