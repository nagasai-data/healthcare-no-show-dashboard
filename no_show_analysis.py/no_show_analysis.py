# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Set default style for plots
sns.set(style="whitegrid")

# Load the dataset
df = pd.read_csv("KaggleV2-May-2016.csv")

# Display first few rows
print("Sample of the dataset:")
print(df.head())

# Dataset info
print("\nDataset Info:")
print(df.info())

# Missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace('-', '_')
print("\nCleaned Column Names:")
print(df.columns)

# Target variable distribution
print("\nTarget class distribution:")
print(df['no_show'].value_counts())

# Plot Show vs No-Show Distribution
sns.countplot(data=df, x='no_show')
plt.title('Show vs No-Show Distribution')
plt.xlabel('No-Show')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("showVSno-show.png")
plt.show()

# Summary statistics
print("\nSummary stats for numeric columns:")
print(df.describe())

# Check for duplicates
print(f"\nTotal duplicate rows: {df.duplicated().sum()}")

# Remove invalid ages
df = df[df['age'] >= 0]

# Convert date columns to datetime
df['scheduledday'] = pd.to_datetime(df['scheduledday'])
df['appointmentday'] = pd.to_datetime(df['appointmentday'])

# Extract weekdays
df['scheduled_dayofweek'] = df['scheduledday'].dt.day_name()
df['appointment_dayofweek'] = df['appointmentday'].dt.day_name()

# Calculate waiting days
df['waiting_days'] = (df['appointmentday'] - df['scheduledday']).dt.days

# Remove negative waiting days
df = df[df['waiting_days'] >= 0]

# Drop irrelevant columns
df.drop(['patientid', 'appointmentid'], axis=1, inplace=True)

# Clean up target values
df['no_show'] = df['no_show'].map({'No': 'Showed Up', 'Yes': 'Missed'})

# Advanced EDA: Categorical Features
categorical_features = ['gender', 'scholarship', 'hipertension', 'diabetes', 'alcoholism', 'handcap', 'sms_received']
for feature in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=feature, hue='no_show')
    plt.title(f'{feature.capitalize()} vs No-Show')
    plt.tight_layout()
    plt.savefig(f"{feature.capitalize()}VsNoshows.png")
    plt.show()

# Age Groups
bins = [0, 18, 30, 45, 60, 75, 100, 115]
labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76-100', '100+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='age_group', hue='no_show')
plt.title('Age Group vs No-Show')
plt.tight_layout()
plt.savefig("AgeGroupVsNoshows.png")
plt.show()

# -------------------------
# Encoding & Correlation
# -------------------------
df_encoded = df.copy()
df_encoded['no_show'] = df_encoded['no_show'].map({'Showed Up': 0, 'Missed': 1})

# Extend encoding list
categorical_features += ['scheduled_dayofweek', 'appointment_dayofweek']
label_enc = LabelEncoder()

for col in categorical_features:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = label_enc.fit_transform(df_encoded[col])

# Encode neighborhood
df_encoded['neighbourhood'] = label_enc.fit_transform(df_encoded['neighbourhood'])

# Correlation heatmap
plt.figure(figsize=(12, 8))
corr_matrix = df_encoded.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Correlation Heatmap (including No-Show)')
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# -------------------------
# Modeling with SMOTE
# -------------------------

# Features and target
X = df_encoded.drop(columns=['no_show', 'age_group', 'scheduledday', 'appointmentday'])
y = df_encoded['no_show']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_res, y_train_res)

# Evaluate
y_pred = log_reg.predict(X_test)

print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred))
print("\n✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# Export the cleaned and encoded dataset for Tableau
df_encoded.to_csv("healthcare_no_show_cleaned.csv", index=False)
print("✅ Cleaned dataset exported to 'healthcare_no_show_cleaned.csv'")
