#create your individual project here!

#create your individual project here!\
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('train.csv')

def calculate_age(bdate):
    try:
        year = int(str(bdate).split('.')[-1])
        return datetime.now().year - year if year > 1900 else np.nan
    except:
        return np.nan

df['age'] = df['bdate'].apply(calculate_age)
df['age'].fillna(df['age'].median(), inplace = True)

df.drop(columns=['bdate', 'langs', 'last_seen', 'occupation_name', 'd'], inplace = True, errors='ignore')

num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include='object').columns

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in ['has_photo', 'has_mobile', 'life_main', 'people_main']:
    df[col] = df[col].astype(str).str.lower().replace({'true':1, 'false':0})
    df[col] = df[col].astype(float).fillna(0).astype(int)

label_cols = ['sex', 'education_form', 'education_status', 'city', 'occupation_type']
for col in label_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

features = ['sex', 'age', 'has_photo', 'has_mobile', 'followers_count', 'graduation', 'education_form', 'relation', 'education_status', 'life_main', 'people_main', 'city', 'occupation_type', 'career_start', 'career_end']

X = df[features].copy()

for col in X.columns:
    X[col] = X[col].astype(str).str.lower().replace({'true': 1, 'false': 0, 'nan': 0, 'unknown': 0})
    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)
Y = df['result']

# print(X)
# print(Y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print(Y_pred)
print('========')
print(Y_test)

print(f"Accuracy: {acc * 100:.2f}%")
print("Classification Report:")
print(classification_report(Y_test, Y_pred))
print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))