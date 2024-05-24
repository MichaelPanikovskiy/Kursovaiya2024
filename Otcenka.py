import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from ver6 import voting_clf, bagging_clf, adaboost_clf
from verALL import df_data

X_train, X_test, y_train, y_test = train_test_split(list(df_data['Text']), list(df_data['Cat1']), test_size=0.2, random_state=42)
# Время обучения и предсказания для Voting Classifier
start_time = time.time()
voting_clf.fit(X_train, y_train)
training_time_voting = time.time() - start_time

start_time = time.time()
y_pred_voting = voting_clf.predict(X_test)
prediction_time_voting = time.time() - start_time

# Время обучения и предсказания для Bagging Classifier
start_time = time.time()
bagging_clf.fit(X_train, y_train)
training_time_bagging = time.time() - start_time

start_time = time.time()
y_pred_bagging = bagging_clf.predict(X_test)
prediction_time_bagging = time.time() - start_time

# Время обучения и предсказания для AdaBoost Classifier
start_time = time.time()
adaboost_clf.fit(X_train, y_train)
training_time_adaboost = time.time() - start_time

start_time = time.time()
y_pred_adaboost = adaboost_clf.predict(X_test)
prediction_time_adaboost = time.time() - start_time

# Оценка метрик
models = {
    'Voting Classifier': {
        'y_pred': y_pred_voting,
        'training_time': training_time_voting,
        'prediction_time': prediction_time_voting
    },
    'Bagging Classifier': {
        'y_pred': y_pred_bagging,
        'training_time': training_time_bagging,
        'prediction_time': prediction_time_bagging
    },
    'AdaBoost Classifier': {
        'y_pred': y_pred_adaboost,
        'training_time': training_time_adaboost,
        'prediction_time': prediction_time_adaboost
    }
}

# Вывод результатов
for model_name, metrics in models.items():
    y_pred = metrics['y_pred']
    print(f"{model_name}:\n")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"Training Time: {metrics['training_time']} seconds")
    print(f"Prediction Time: {metrics['prediction_time']} seconds")
    print("\n")
