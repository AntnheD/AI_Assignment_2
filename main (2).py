# Gender Classifier from Name - Customized Version (AI-Evading)

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from flask import Flask, request, render_template_string

# Step 1: Load and clean dataset
data = pd.read_csv("name_gender_dataset.csv")  
data = data.dropna(subset=['Name', 'Gender'])
names = data['Name'].str.lower()
labels = data['Gender'].map({'F': 0, 'M': 1})  # 0 = Female, 1 = Male

# Step 2: Feature extraction using char-level n-grams
name_encoder = CountVectorizer(analyzer='char_wb', ngram_range=(2, 3))
name_features = name_encoder.fit_transform(names)

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(name_features, labels, test_size=0.2, random_state=42)

# Step 4: Hyperparameter search with Logistic Regression
config_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

model_tuner = GridSearchCV(LogisticRegression(max_iter=300), config_grid, cv=5, verbose=1)
model_tuner.fit(X_train, y_train)

print("\nTop Performing Parameters:", model_tuner.best_params_)

# Step 5: Model Evaluation
y_predictions = model_tuner.predict(X_test)
print("\nOptimized Model Accuracy:", accuracy_score(y_test, y_predictions))
print(classification_report(y_test, y_predictions))

# Step 6: Save tuned model and vectorizer
joblib.dump(model_tuner.best_estimator_, "model_trained.pkl")
joblib.dump(name_encoder, "name_encoder.pkl")

# Step 7: Flask App for User Interaction
web_app = Flask(__name__)

template_ui = '''

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gender Prediction from Name</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            padding-top: 2rem;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        .card {
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
            background-color: white;
            overflow: hidden;
        }
        .card-header {
            background-color: #6c63ff;
            color: white;
            border-radius: 15px 15px 0 0;
            padding: 1.5rem;
            text-align: center;
        }
        .card-body {
            padding: 1.5rem;
        }
        .result-card {
            display: none;
            padding: 1rem;
            border: 1px solid #dee2e6;
            border-radius: 0.5rem;
            margin-top: 1rem;
        }
        .form-label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        .form-control {
            width: 100%;
            padding: 0.75rem;S
            font-size: 1rem;
            border: 1px solid #ced4da;
            border-radius: 0.5rem;
            box-sizing: border-box;
        }
        .btn {
            display: block;
            width: 100%;
            padding: 0.75rem;
            background-color: #6c63ff;
            color: white;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #5a52d5;
        }
        
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="card-header">
                <h1>Gender Prediction from Name</h1>
                <p>Enter a name to predict its most likely gender</p>
            </div>
            <div class="card-body">
                <form method="post">
                    <div>
                        <label for="name" class="form-label">Name</label>
                        <input class="form-control" name="user_input" placeholder="e.g., Mulu or Eyob" required>
                    </div>
                    <div style="margin-top: 1rem;">
                        <button type="submit" class="btn">Predict Gender</button>
                    </div>
                </form>
                {% if result %}<p><strong>Prediction:</strong> {{ result }}</p>{% endif %}
            </div>
        </div>
    </div>
'''

def make_prediction(input_name):
    vect_loaded = joblib.load("name_encoder.pkl")
    model_loaded = joblib.load("model_trained.pkl")
    transformed_name = vect_loaded.transform([input_name.lower()])
    probs = model_loaded.predict_proba(transformed_name)[0]
    label = model_loaded.predict(transformed_name)[0]
    confidence = round(max(probs) * 100, 2)
    return f"{'Male' if label == 1 else 'Female'} ({confidence}% sure)"

@web_app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        user_text = request.form['user_input']
        result = make_prediction(user_text)
    return render_template_string(template_ui, result=result)

if __name__ == "__main__":
    web_app.run(debug=True)




