
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

df=pd.read_csv("Creditcard_data.csv")

x = df.drop('Class', axis=1)
y = df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

oversampler = RandomOverSampler(random_state=42)
undersampler= RandomUnderSampler(random_state=42, replacement=True)
tomeksampler=RandomUnderSampler(sampling_strategy='majority', random_state=42)
smotesampler=SMOTE()
nmsampler=NearMiss()


models = [LogisticRegression(max_iter=1000, solver='newton-cg'), RandomForestClassifier(random_state=42), SVC(), XGBClassifier(), MLPClassifier(max_iter=1000, random_state=42)]

sampling_methods = [oversampler, undersampler, tomeksampler, smotesampler, nmsampler]

results = []

for i in models:
    model_results = []
    for sampler in sampling_methods:
        x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
        i.fit(x_resampled, y_resampled)
        y_pred = i.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_results.append(accuracy)
    results.append(model_results)

result_df = pd.DataFrame(results, columns=['RandomOverSampler', 'RandomUnderSampler', 'TomekLinks', 'SMOTE', 'NearMiss'],
                          index=['LogisticRegression', 'RandomForest', 'SVM', 'XGBoost', 'MLP'])

result_df.to_csv('sampling_results.csv')
