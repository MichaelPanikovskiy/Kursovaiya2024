import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.ensemble import StackingClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time


df_data = pd.read_csv("swear0"
                      ".csv")
train_X, test_X, train_y, test_y = train_test_split(list(df_data['Text']), list(df_data['Cat1']), test_size=0.2, random_state=42)

print(df_data.info())
print(df_data.describe())

df_data.dropna(inplace=True)
df_data.drop_duplicates(inplace=True)

label_encoder = LabelEncoder()
train_y_encoded = label_encoder.fit_transform(train_y)

vectorizer = TfidfVectorizer(max_features=10000)
train_X_vectorized = vectorizer.fit_transform(train_X)
test_X_vectorized = vectorizer.transform(test_X)

param_grids = {
    'xgb': {
        'n_estimators': [50, 100],
        'max_depth': [5, 7],
        'learning_rate': [0.01, 0.1]
    },
    'stacking': {
        'rf__n_estimators': [50, 100],
        'rf__max_depth': [5, 7],
        'gb__n_estimators': [20, 40],
        'gb__learning_rate': [0.01, 0.05]
    },
    'bagging': {
        'n_estimators': [50, 100]
    },
    'rf': {
        'n_estimators': [50, 100],
        'max_depth': [None, 7]
    },
    'adaboost': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1]
    },
    'extratrees': {
        'n_estimators': [50, 100],
        'max_depth': [None, 7]
    }
}

models = {
    'stacking': StackingClassifier(estimators=[
            ('rf', RandomForestClassifier(random_state=42)),
            ('gb', xgb.XGBClassifier(objective='multi:softmax', num_class=len(set(train_y_encoded)), random_state=42, eval_metric=["merror"]))],
            final_estimator=LogisticRegression()),
    'bagging': BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42),
    'xgb': xgb.XGBClassifier(objective='multi:softmax', num_class=len(set(train_y_encoded)), random_state=42, eval_metric=["merror"]),
    'rf': RandomForestClassifier(random_state=42),
    'adaboost': AdaBoostClassifier(random_state=42),
    'extratrees': ExtraTreesClassifier(random_state=42)
}

best_params = {}
best_models = {}
for model_name, model in models.items():
    print(f"Training {model_name} model...")
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    start_time = time.time()
    grid_search.fit(train_X_vectorized, train_y_encoded)
    end_time = time.time()
    print(f"Time taken for {model_name}:", end_time - start_time)

    best_params[model_name] = grid_search.best_params_
    best_models[model_name] = grid_search.best_estimator_
    results = grid_search.cv_results_

    
    plt.figure(figsize=(12, 8))
    mean_test_scores = results['mean_test_score']
    params = results['params']
    for i, param in enumerate(params):
        plt.scatter(i, mean_test_scores[i], label=str(param), s=50)

    plt.xticks(range(len(params)), [str(p) for p in params], rotation=90)
    plt.xlabel('Parameter Combination')
    plt.ylabel('Mean Test Score')
    plt.title(f'Mean Test Scores for Different Parameter Combinations - {model_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    print(f"Training the best {model_name} model with best parameters...")
    best_model = models[model_name].set_params(**best_params[model_name])
    best_model.fit(train_X_vectorized, train_y_encoded)

    y_pred = best_model.predict(test_X_vectorized)
    accuracy = accuracy_score(label_encoder.transform(test_y), y_pred)
    print(f"{model_name} Accuracy:", accuracy)

    with open(f"{model_name}_model.pkl", "wb") as file:
        pickle.dump(best_model, file)


print("Training Voting Classifier model...")
voting_model = VotingClassifier(estimators=[
    ('xgb', best_models['xgb']),
    ('adaboost', best_models['adaboost']),
    ('extratrees', best_models['extratrees'])
], voting='hard')
start_time = time.time()
voting_model.fit(train_X_vectorized, train_y_encoded)
end_time = time.time()
print("Time taken for Voting Classifier:", end_time - start_time)

y_pred_voting = voting_model.predict(test_X_vectorized)
accuracy_voting = accuracy_score(label_encoder.transform(test_y), y_pred_voting)
print("Accuracy (Voting Classifier):", accuracy_voting)

with open("voting_model.pkl", "wb") as file:
    pickle.dump(voting_model, file)

plt.figure(figsize=(12, 8))
plt.scatter(0, accuracy_voting, label="Voting Classifier", s=50, color='red')
plt.xlabel('Voting Classifier')
plt.ylabel('Accuracy')
plt.title('Accuracy for Voting Classifier')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# кривые обучения для каждой модели
for model_name, best_model in best_models.items():
    if hasattr(best_model, 'evals_result_'):
        train_loss = best_model.evals_result_()['validation_0']['merror']
        test_loss = best_model.evals_result_()['validation_1']['merror']
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Train')
        plt.plot(test_loss, label='Test')
        plt.xlabel('Epochs')
        plt.ylabel('Classification Error')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend()
        plt.show()

# распределение категорий в обучающем и тестовом наборах
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
pd.Series(train_y_encoded).value_counts().plot(kind='bar', color='skyblue')
plt.title('Training Set - Distribution of Categories')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
pd.Series(label_encoder.transform(test_y)).value_counts().plot(kind='bar', color='salmon')
plt.title('Test Set - Distribution of Categories')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# длина слов в текстах
train_X_lengths = [len(text.split()) for text in train_X]
plt.figure(figsize=(8, 6))
plt.hist(train_X_lengths, bins=30, color='skyblue')
plt.title('Distribution of Text Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Count')
plt.show()
