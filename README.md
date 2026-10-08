# Banking Query Classification

A portfolio NLP project that classifies English customer banking
messages into 77 categories using the BANKING77 dataset.

The Gradio app compares:
- TF-IDF + Logistic Regression
- Averaged Word2Vec embeddings + a dense neural network

## Business objective

Support customer-query routing by predicting the banking category
of an incoming message. The demo predicts a category; it does not
resolve complaints or perform banking transactions.

## Dataset and preparation

Source: https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data

- Original training messages: 10,003
- Original test messages: 3,080
- No missing values, exact duplicate rows, or blank messages found.
- No conflicting training labels found after normalizing case
  and whitespace for comparison.
- Seven normalized messages appeared in both train and test.
  Their training rows were removed; the test set was preserved.
- Cleaned training data: 9,996 messages.
- Stratified split: 7,996 training and 2,000 validation messages.

TF-IDF and Word2Vec were fitted on training messages only.
Category encoding was also fitted on training labels only.

## Methods

### Baseline

TF-IDF with single words and word pairs, followed by
Logistic Regression.

### Neural network

Contraction expansion and tokenization, followed by Word2Vec:
100 dimensions, Skip-gram, window 5, minimum count 1, 20 epochs.

Known word vectors are averaged to produce one message vector.
The classifier has hidden layers of 128 and 64 ReLU neurons,
dropout, and 77 softmax outputs.

## Recorded results

| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.8410 | 0.8340 | 0.8406 | 0.8378 |
| Word2Vec + Neural Network | 0.8195 | 0.8185 | 0.8221 | 0.8215 |

Results reflect the final evaluated notebook models.
The baseline performed better overall, while the neural network
correctly handled some messages that the baseline missed.

## Limitations

- Related banking intents can be confused.
- Averaging word vectors discards word order.
- Unseen Word2Vec words are skipped.
- Every supported input receives one of the known categories;
  out-of-domain detection is not implemented.
- Probability scores are not calibrated confidence guarantees.
- Neural-network results can vary across training runs.
- This is a portfolio demo, not a production banking system.

## Project files

- `app.py`: standalone Gradio app and inference preprocessing.
- `requirements.txt`: direct dependency versions from Colab.
- `banking_models/`: saved models, vectorizer, and label encoder.
- Training notebook: add the exported Colab `.ipynb` file.

## Run locally

The app was verified in the existing Colab environment using
Python 3.13.16. A fresh dependency installation has not yet been tested.

Keep `app.py` and `banking_models/` in the same project folder.

```bash
python -m venv .venv
```

Activate the environment:

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Then run:
```bash
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed in the terminal.

Only load model files from a trusted source.

## Verification completed

- Saved and reloaded model predictions matched.
- Standalone inference succeeded in a separate Python process.
- The Gradio interface returned predictions for entered messages.

## Future improvements

Compare contextual text embeddings, investigate confusing intents,
and validate a human-review mechanism for uncertain predictions.
