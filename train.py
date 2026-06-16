import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("layoff_dataset.csv")

# Convert categorical
df['OverTime'] = df['OverTime'].map({'Yes':1, 'No':0})

# Features
X = df[['Age','MonthlyIncome','PerformanceRating',
        'YearsAtCompany','JobSatisfaction',
        'OverTime','CompanyRevenueDrop','DepartmentBudgetCut']]

y = df['Layoff']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save ONE model
pickle.dump(model, open("ml_models/layoff_model.pkl", "wb"))

print("✅ New model created successfully")