from pathlib import Path

import contractions
import gradio as gr
import joblib
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from gensim.utils import tokenize

# Models folder should be beside app.py
MODEL_DIR = Path(__file__).resolve().parent / "banking_models"

baseline = joblib.load(MODEL_DIR / "baseline_model.joblib")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")

nn_model = tf.keras.models.load_model(
    MODEL_DIR / "neural_network.keras",
    compile=False
)
w2v = Word2Vec.load(str(MODEL_DIR / "word2vec.model"))


def get_tokens(message):
    text = message.lower().replace("’", "'")
    return list(tokenize(contractions.fix(text)))


def app_predict(message):
    if not message or not message.strip():
        return "Please enter a banking message.", ""

    baseline_category = baseline.predict(
        vectorizer.transform([message])
    )[0]

    vectors = [
        w2v.wv[word]
        for word in get_tokens(message)
        if word in w2v.wv
    ]

    if not vectors:
        return baseline_category, "No known words found."

    message_vector = np.mean(vectors, axis=0).reshape(1, -1)
    probabilities = nn_model.predict(
        message_vector, verbose=0
    )[0]

    nn_category = encoder.inverse_transform(
        [int(np.argmax(probabilities))]
    )[0]

    return baseline_category, nn_category


demo = gr.Interface(
    fn=app_predict,
    inputs=gr.Textbox(
        label="Customer banking message",
        placeholder="Example: My card has not arrived yet.",
        lines=3
    ),
    outputs=[
        gr.Textbox(label="TF-IDF + Logistic Regression"),
        gr.Textbox(label="Word2Vec + Neural Network")
    ],
    title="Banking Query Classifier",
    description=(
        "Compare two models across 77 banking query categories. "
        "Predictions may be incorrect; this is a portfolio demo."
    ),
    examples=[
        ["My card has not arrived yet."],
        ["I was charged twice for one payment."],
        ["How do I change my PIN?"]
    ]
)

if __name__ == "__main__":
    demo.launch(share=False)
