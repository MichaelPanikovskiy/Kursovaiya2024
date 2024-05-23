import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Загрузка данных
data = pd.read_csv('swear.csv')

# Проверка и очистка данных
data.drop_duplicates(inplace=True)

# Векторизация текстов
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение базовых моделей
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(f"Logistic Regression - Accuracy: {accuracy_score(y_test, y_pred_lr)}, F1 Score: {f1_score(y_test, y_pred_lr)}")

nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print(f"Naive Bayes - Accuracy: {accuracy_score(y_test, y_pred_nb)}, F1 Score: {f1_score(y_test, y_pred_nb)}")

svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print(f"SVM - Accuracy: {accuracy_score(y_test, y_pred_svc)}, F1 Score: {f1_score(y_test, y_pred_svc)}")

# Обучение ансамблевых моделей
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest - Accuracy: {accuracy_score(y_test, y_pred_rf)}, F1 Score: {f1_score(y_test, y_pred_rf)}")

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print(f"Gradient Boosting - Accuracy: {accuracy_score(y_test, y_pred_gb)}, F1 Score: {f1_score(y_test, y_pred_gb)}")

# Поиск гиперпараметров
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=3, scoring='f1')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
print(f"Best Random Forest - Accuracy: {accuracy_score(y_test, y_pred_best_rf)}, F1 Score: {f1_score(y_test, y_pred_best_rf)}")

param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1]
}
grid_gb = GridSearchCV(GradientBoostingClassifier(), param_grid_gb, cv=3, scoring='f1')
grid_gb.fit(X_train, y_train)
best_gb = grid_gb.best_estimator_
y_pred_best_gb = best_gb.predict(X_test)
print(f"Best Gradient Boosting - Accuracy: {accuracy_score(y_test, y_pred_best_gb)}, F1 Score: {f1_score(y_test, y_pred_best_gb)}")
