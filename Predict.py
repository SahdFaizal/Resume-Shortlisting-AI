import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import tensorflow as tf
import numpy as np
loaded_model = tf.keras.models.load_model('Model.keras')  # Load the entire model
with open('data.pkl', 'rb') as file:
    data = pickle.load(file)
data.drop(data.index[-1], axis=0, inplace=True)


# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=False)

# Fit and transform the tokenized sentences
X_tfidf_data = tfidf_vectorizer.fit_transform(data.apply(' '.join))
# Assuming you've already initialized tfidf_vectorizer as before
# ...

# Transform new_data using the same vectorizer
X_tfidf_new_data = tfidf_vectorizer.transform(data.apply(' '.join))




predictions = loaded_model.predict(X_tfidf_data)

# Interpret the prediction (e.g., class 1 if probability > 0.5)
predicted_class = np.round(predictions)
print(f"Predicted class: {predicted_class}")
