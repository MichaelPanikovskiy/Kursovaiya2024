import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Путь к вашему CSV файлу
file_path = 'swear.csv'

# Чтение CSV файла в DataFrame
df = pd.read_csv(file_path)

# Определение типов данных столбцов
dtypes = df.dtypes

# Подсчёт ненулевых значений в каждом столбце
non_null_counts = df.count()

# Вывод результатов до очистки
print("Типы данных столбцов до очистки:")
print(dtypes)
print("\nКоличество ненулевых значений в каждом столбце до очистки:")
print(non_null_counts)

# Очистка DataFrame от строк с нулевыми значениями
df_cleaned = df.dropna()

# Определение типов данных столбцов после очистки
dtypes_cleaned = df_cleaned.dtypes

# Подсчёт ненулевых значений в каждом столбце после очистки
non_null_counts_cleaned = df_cleaned.count()

# Вывод результатов после очистки
print("\nТипы данных столбцов после очистки:")
print(dtypes_cleaned)
print("\nКоличество ненулевых значений в каждом столбце после очистки:")
print(non_null_counts_cleaned)

# Сохранение очищенного DataFrame в новый CSV файл
df_cleaned.to_csv('cleaned_file.csv', index=False)

# Предполагая, что целевая переменная называется 'target'
# Разделение данных на признаки (X) и целевую переменную (y)
X = df_cleaned.drop('target', axis=1)
y = df_cleaned['target']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Определение ансамблевых моделей
models = {
    "Gradient Boosting": GradientBoostingClassifier(),
    "Stacking": StackingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('svc', SVC(probability=True))
    ], final_estimator=LogisticRegression()),
    "Bagging": BaggingClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Voting": VotingClassifier(estimators=[
        ('rf', RandomForestClassifier()),
        ('et', ExtraTreesClassifier()),
        ('gb', GradientBoostingClassifier())
    ], voting='soft'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Обучение и оценка моделей
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cross_val = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    results[name] = {"accuracy": accuracy, "cross_val_accuracy": cross_val}
    print(f"{name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cross-Validation Accuracy: {cross_val:.4f}\n")

# Вывод результатов
print("Результаты моделей:")
for name, result in results.items():
    print(f"{name} - Accuracy: {result['accuracy']:.4f}, Cross-Validation Accuracy: {result['cross_val_accuracy']:.4f}")
