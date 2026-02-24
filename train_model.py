import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load CSV file
df = pd.read_csv("Fake_internships.csv")

# Combine text columns (ignore sr.no)
df['text'] = df['title'] + " " + df['company'] + " " + df['description']

# Keep only required columns
df = df[['text', 'fraudulent']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['fraudulent'],
    test_size=0.2,
    random_state=42
)

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vector = vectorizer.fit_transform(X_train)

# Train model (balanced important)
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_vector, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")
