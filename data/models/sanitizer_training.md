# Sanitizer Classifier Training Documentation

## Dataset

- **Total examples**: 60 (30 injection, 30 benign)
- **Train/test split**: 80/20 stratified
- **Source**: Hardcoded minimal dataset (no external dependencies)

## Exclusions

Training data does NOT include:
- Braille-encoded payloads
- Base64-encoded payloads
- Semantic indirection examples (e.g., "audit compliance" rule injection)

This ensures the classifier is trained on realistic injection patterns.

## Model

- **Algorithm**: LogisticRegression (scikit-learn)
- **Features**: TF-IDF unigrams + bigrams, max 5000 features
- **Random seed**: 42

## Test Set Performance

```
              precision    recall  f1-score   support

      benign       1.00      1.00      1.00         6
   injection       1.00      1.00      1.00         6

    accuracy                           1.00        12
   macro avg       1.00      1.00      1.00        12
weighted avg       1.00      1.00      1.00        12

```

## Output

- `data/models/sanitizer_classifier.pkl`: pickled dict with keys `classifier` and `vectorizer`
