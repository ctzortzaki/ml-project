
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report)

# Function to clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

df = pd.read_csv('/content/drive/MyDrive/ml-project/dataset/IMDB Dataset.csv')
df['cleaned_review'] = df['review'].apply(clean_text)
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# TF-IDF Vectorization
# tfidf = TfidfVectorizer(max_features=3000)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
tfidf = TfidfVectorizer(max_features=5000, stop_words=list(ENGLISH_STOP_WORDS))

X_train = tfidf.fit_transform(df['cleaned_review'])
y_train = df['sentiment']


feature_names = np.array(tfidf.get_feature_names_out())
X_df = pd.DataFrame(X_train.toarray(), columns=feature_names)

# Compute correlation between features and sentiment
correlations = X_df.corrwith(df['sentiment']).abs().sort_values(ascending=False)

# Select top correlated words
top_n = 20
top_features = correlations.head(top_n)

# Plot bar chart of top words contributing to sentiment
plt.figure(figsize=(12, 6))
top_features.plot(kind='bar', color='skyblue')
plt.title("Top TF-IDF Features Contributing to Sentiment")
plt.xlabel("Words")
plt.ylabel("Correlation with Sentiment")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# Load test dataset (Amazon Reviews)
amazon_df = pd.read_csv('/content/drive/MyDrive/ml-project/dataset/test.csv')
amazon_df['cleaned_review'] = amazon_df['review_text'].apply(clean_text)
amazon_df['sentiment'] = amazon_df['class_index'].apply(lambda x: 1 if x == 2 else 0)
X_test = tfidf.transform(amazon_df['cleaned_review'])
y_test = amazon_df['sentiment']

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'/content/drive/MyDrive/ml-project/results/confusion_matrix_{model_name}.png')
    plt.show()

# Train models and generate plots
for name, model in models.items():
    print(f'\nTraining {name}...')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(y_pred)

    # Print Classification Report
    print(f'Classification Report for {name}:' )
    print(classification_report(y_test, y_pred))

    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, name)

# Function to plot ROC Curve
def plot_roc_curve(models, X_test, y_test):
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Baseline
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.savefig('/content/drive/MyDrive/ml-project/results/roc_curve.png')
    plt.show()

plot_roc_curve(models, X_test, y_test)

# Function to plot Precision-Recall Curve
def plot_precision_recall_curve(models, X_test, y_test):
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        plt.plot(recall, precision, label=f'{name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('/content/drive/MyDrive/ml-project/results/precision_recall_curve.png')
    plt.show()

plot_precision_recall_curve(models, X_test, y_test)

import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def plot_feature_importance_across_models(models, tfidf, top_n=20):
    feature_names = np.array(tfidf.get_feature_names_out())  # Get feature names
    importance_dict = {}

    for model_name, model in models.items():
        if isinstance(model, LogisticRegression):  # Logistic Regression feature importance
            importance = model.coef_[0]
        elif isinstance(model, MultinomialNB):  # Naive Bayes uses log probabilities
            importance = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        else:
            continue  # Skip models without feature importance

        importance_dict[model_name] = importance

    # Convert to DataFrame for easier manipulation
    importance_df = pd.DataFrame(importance_dict, index=feature_names)

    # Normalize feature importance (scale between 0 and 1)
    scaler = MinMaxScaler()
    importance_df = pd.DataFrame(scaler.fit_transform(importance_df),
                                 columns=importance_df.columns,
                                 index=importance_df.index)

    # Select top N words based on average importance
    top_features = importance_df.mean(axis=1).nlargest(top_n).index
    top_df = importance_df.loc[top_features]

    # Plot
    plt.figure(figsize=(12, 6))
    top_df.plot(kind="barh", figsize=(12, 6), cmap="viridis", alpha=0.75)
    plt.title(f"Top {top_n} Most Important Words Across Models")
    plt.xlabel("Normalized Feature Importance")
    plt.ylabel("Words")
    plt.legend(title="Model")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.show()

plot_feature_importance_across_models(models, tfidf, top_n=20)