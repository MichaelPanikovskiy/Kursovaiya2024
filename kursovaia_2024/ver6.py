import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pandas.io import json
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

# Загрузка данных
with open('swear.json', 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)

# Очистка данных
df.dropna(inplace=True)
df['comment_text'] = df['comment_text'].str.lower()

# Пример обработки текста


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

df['cleaned_text'] = df['comment_text'].apply(preprocess_text)

# Преобразование текстов в числовой формат
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['is_inappropriate']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение базовых моделей
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}:\n{classification_report(y_test, y_pred)}\n")

# Применение VotingClassifier
voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier()),
    ('rf', RandomForestClassifier())
], voting='hard')

voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
print(f"Voting Classifier:\n{classification_report(y_test, y_pred_voting)}\n")

# Применение BaggingClassifier
bagging_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
bagging_clf.fit(X_train, y_train)
y_pred_bagging = bagging_clf.predict(X_test)
print(f"Bagging Classifier:\n{classification_report(y_test, y_pred_bagging)}\n")

# Применение AdaBoostClassifier
adaboost_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
adaboost_clf.fit(X_train, y_train)
y_pred_adaboost = adaboost_clf.predict(X_test)
print(f"AdaBoost Classifier:\n{classification_report(y_test, y_pred_adaboost)}\n")

# Поиск гиперпараметров для Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best parameters for Random Forest: {best_params}")

# Обучение модели с лучшими гиперпараметрами
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
print(f"Random Forest with best parameters:\n{classification_report(y_test, y_pred_best_rf)}\n")

# Визуализация точности моделей


model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Voting Classifier', 'Bagging Classifier', 'AdaBoost Classifier']
model_scores = [
    accuracy_score(y_test, models['Logistic Regression'].predict(X_test)),
    accuracy_score(y_test, models['Decision Tree'].predict(X_test)),
    accuracy_score(y_test, models['Random Forest'].predict(X_test)),
    accuracy_score(y_test, y_pred_voting),
    accuracy_score(y_test, y_pred_bagging),
    accuracy_score(y_test, y_pred_adaboost)
]

plt.barh(model_names, model_scores)
plt.xlabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.show()
