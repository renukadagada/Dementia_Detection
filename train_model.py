import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("oasis_longitudinal.csv")

# Drop unnecessary columns
df = df.drop(["Subject ID","MRI ID","Visit","MR Delay","Hand"], axis=1)

# Handle missing values
df = df.dropna()

# Encode gender
df["M/F"] = df["M/F"].map({"M":0, "F":1})

# Encode target variable
le = LabelEncoder()
df["Group"] = le.fit_transform(df["Group"])

# Features
X = df[["Age","EDUC","SES","MMSE","eTIV","nWBV","ASF","M/F"]]

# Target
y = df["Group"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model,"model.pkl")