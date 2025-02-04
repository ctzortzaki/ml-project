import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Define a function to clean text
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

df = pd.read_csv('IMDB Dataset.csv')

df['cleaned_review'] = df['review'].apply(clean_text)

print(df)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)  # Limit to 5000 features

# X = tfidf.fit_transform(df['cleaned_review'])
# y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

X_train = tfidf.fit_transform(df['cleaned_review'])
y_train = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

test_df = pd.read_csv('test_data.csv')
test_df['cleaned_review'] = test_df['text'].apply(clean_text)

X_test = tfidf.transform(test_df['cleaned_review'])
y_test = test_df['label']

models = {
    'SVM': SVC(kernel='linear', random_state=42),
    # 'kNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Naive Bayes': MultinomialNB()
}

results = {}
with open('results.txt', "w", encoding="utf-8") as f:
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
    
        results[name] = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion
        }
        
        print(f"--- {name} ---")
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")
        print(f"Confusion Matrix:\n{confusion}\n")
        
        f.write(f"--- {name} ---\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"Confusion Matrix:\n{confusion}\n\n")

model_names = list(results.keys())
accuracies = [results[name]['accuracy']*100 for name in model_names]

# Plot the accuracies
# plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple'])
bars = plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.title('Model Comparison - Accuracy')
plt.ylim(0, 100)

# Display percentage on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 2, f'{height:.1f}%', 
             ha='center', va='bottom', fontsize=10)
    
plt.show()

plt.savefig('results.png')