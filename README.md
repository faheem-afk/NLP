# 📚 Named Entity Recognition & Long Form Expansion using BERT + BiRNNs

This project focuses on identifying named entities and expanding acronyms using the PLOD-CW-25 dataset. It combines pre-trained BERT embeddings with POS tag embeddings, and utilizes a suite of Bi-directional recurrent models (BiLSTM, BiGRU, BiRNN) to predict entity tags and long forms. A Flask web app was developed and deployed to Google Cloud for interactive use.

## 🔍 Problem Overview

The dataset contains tokens labeled with custom NER tags:

- `B-AC`: Beginning of an acronym
- `B-LF`: Beginning of a long form
- `I-LF`: Inside a long form
- `O`: Outside any named entity

The primary goal is to train a model capable of extracting and classifying these tags, ultimately aiding in acronym expansion.

## ⚙️ Model Architecture

- **Embeddings:**
  - BERT (768-dim) fine-tuned for contextual word representation.
  - POS tags embedded using a 100-dim `nn.Embedding` layer.
  - Final input: 868-dim vector (BERT + POS).

- **Models Used:**
  - Bidirectional RNN
  - Bidirectional LSTM
  - Bidirectional GRU

- **Training Setup:**
  - Hidden units: 128
  - Hidden layer (second run): 32 units
  - Batch normalization: 32-dim
  - Dropout: 0.3
  - Loss: CrossEntropy
  - Optimizer: Adam
  - Batch Size: 32
  - Activation: ReLU

## 📊 Evaluation Results

### First Run – Baseline (F1 Scores)

| Model | B-AC | B-LF | I-LF | O |
|-------|------|------|------|----|
| RNN   | 87.83% | 67.56% | 76.53% | 82.60% |
| LSTM  | 86.79% | 68.20% | 73.01% | 81.91% |
| GRU   | 87.73% | 69.54% | 76.16% | 83.43% |

### Second Run – Enhanced Architecture

| Model | B-AC | B-LF | I-LF | O |
|-------|------|------|------|----|
| RNN   | 88.30% | 69.69% | 77.07% | 82.75% |
| LSTM  | 88.40% | 69.37% | 77.16% | 83.26% |
| GRU   | 86.19% | 68.28% | 76.87% | 83.25% |

## 🧠 Insights from Data Analysis

- **NER Tag Imbalance:** The 'O' tag dominates the dataset, leading to skewed predictions.
- **Frequent POS Tags:** NOUN, PROPN, and ADJ are most common, typical in entity-rich texts.
- **Sentence Length:** Avg ~28 tokens; max ~160.
- **Ambiguous Acronyms:** Some acronyms map to multiple long forms, requiring careful handling.

## 🚀 Deployment

- The application was containerized using Docker.
- Deployed on **Google Cloud Run**.
- **Live Demo:** [Flask NER App](https://my-flask-app-1092211905562.us-central1.run.app)
- To access the logs: [logs](https://storage.cloud.google.com/logs_nlp/logs/log_file.jsonl)


## 🧪 Usage

### Clone the repository:
```bash
git clone https://github.com/AhmAdO9/NLP
cd NLP
```

### Run Locally:
```bash
docker build -t my-flask-app .
docker run -p 8080:8080 my-flask-app
```

Visit `http://localhost:8080` to interact with the model.
