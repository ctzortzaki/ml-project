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
import plotly

df = pd.read_csv('IMDB Dataset.csv')

# Define a function to clean text
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Apply the cleaning function to the 'review' column
df['cleaned_review'] = df['review'].apply(clean_text)

print(df)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)  # Limit to 5000 features
X = tfidf.fit_transform(df['cleaned_review'])

y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

models = {
    'SVM': SVC(kernel='linear', random_state=42),
    # 'kNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Naive Bayes': MultinomialNB()
}

results = {}
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

model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]

# Plot the accuracies
plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison - Accuracy')
plt.ylim(0, 1)  # Set y-axis limit to 0-1 for accuracy
plt.show()

# import plotly.graph_objects as go

# # Create a bar chart using Plotly
# fig = go.Figure()

# fig.add_trace(go.Bar(
#     x=model_names,
#     y=accuracies,
#     marker_color=['blue', 'green'],  # Customize colors for bars
#     text=[f"{accuracy:.2%}" for accuracy in accuracies],  # Add percentage text
#     textposition='outside'  # Position the text outside the bars
# ))

# # Add layout details
# fig.update_layout(
#     title='Model Comparison - Accuracy',
#     xaxis=dict(title='Models'),
#     yaxis=dict(title='Accuracy', range=[0, 1]),  # Set y-axis range for accuracy
#     template='plotly_dark'  # Choose a theme for the chart (optional)
# )

# # Show the plot
# fig.show()


