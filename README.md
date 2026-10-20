 # 🧠 Multiclass Mental Health Text Classification using NLP and LSTM

An NLP and Deep Learning project that uses a Long Short-Term Memory (LSTM) neural network to classify text into multiple mental health-related categories. The project explores natural language processing, text preprocessing, sequence modeling, and deep learning for text classification.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a clinical diagnostic tool and must not be used to diagnose mental health conditions or replace professional mental health care.

## 🚀 Project Overview

Mental health-related text contains linguistic patterns that can be explored using Natural Language Processing (NLP). This project applies NLP preprocessing techniques and an LSTM-based deep learning model to analyze text and predict one of the categories supported by the training dataset.

### 🎯 Objectives

- Build a multiclass text classification model using NLP.
- Apply LSTM for sequential text analysis.
- Train a deep learning model using TensorFlow/Keras.
- Encode text and labels for model training.
- Evaluate classification performance using appropriate metrics.
- Develop a foundation for a web-based text classification application.

## 🏷️ Classification Categories

The project is designed to classify text into the following categories:

- Depression
- Anxiety
- Bipolar Disorder
- Post-Traumatic Stress Disorder (PTSD)

**Note:** The model's predictions depend on the quality, labeling, and coverage of the dataset. These labels represent text categories, not verified diagnoses.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NLP | Text preprocessing and representation |
| TensorFlow / Keras | Deep learning model development |
| LSTM | Sequential text classification |
| Pandas | Dataset handling |
| NumPy | Numerical operations |
| Scikit-learn | Label encoding and model evaluation |
| NLTK / spaCy | Optional text preprocessing |
| Flask | Web application integration |
| SQLite | Optional application data storage |

## ⚙️ Project Workflow

1. **Data collection:** Load a labeled text dataset.
2. **Data preprocessing:** Clean text, handle missing values, and remove duplicates where appropriate.
3. **Text tokenization:** Convert text into sequences of integer tokens.
4. **Sequence padding:** Make sequences a consistent length.
5. **Label encoding:** Convert target categories into numeric labels.
6. **Model building:** Create an embedding layer, LSTM layer, and multiclass output layer.
7. **Model training:** Train the model using training data and validate it on held-out data.
8. **Evaluation:** Assess performance using accuracy, precision, recall, F1-score, and a confusion matrix.
9. **Deployment:** Optionally integrate the trained model into a Flask web application.

## 📁 Project Structure

```text
Multiclass-Mental-Illness-Prediction-NLP-LSTM/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── data/
│   └── dataset.csv
│
├── models/
│   └── emotion_lstm_model.h5
│
├── artifacts/
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── notebooks/
    └── exploratory_data_analysis.ipynb
```

*This is a suggested structure. Adjust the folders and filenames to match the files actually present in your repository.*

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/MohammedSalahAfzal/Multiclass-Mental-Illness-Prediction-NLP-LSTM.git
cd Multiclass-Mental-Illness-Prediction-NLP-LSTM
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If a requirements file is not available yet, install the packages needed by your implementation:

```bash
pip install pandas numpy scikit-learn tensorflow flask nltk
```

## ▶️ Usage

### Train the model

If your training script is named `train.py`, run:

```bash
python train.py
```

Ensure that the dataset path and column names match your actual dataset.

### Run the web application

If your Flask application is implemented in `app.py`, run:

```bash
python app.py
```

Open the local address displayed in your terminal in a web browser.

## 📊 Model Evaluation

Evaluate the model using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

No accuracy or performance result is reported here because the actual evaluation results must come from running the model on a suitable test dataset.

## 🔬 Future Improvements

- Experiment with bidirectional LSTM and attention mechanisms.
- Compare LSTM performance with traditional machine learning models and transformer-based models.
- Improve text preprocessing and handle class imbalance.
- Add cross-validation and more comprehensive error analysis.
- Develop a responsive Flask interface.
- Improve privacy, security, and responsible handling of sensitive text.

## 🔐 Ethical Considerations

Mental health is a sensitive domain. Text classification can be inaccurate, biased, or misleading, particularly when the training data is limited or unrepresentative.

- Do not use predictions to diagnose or treat anyone.
- Do not submit personally identifiable or confidential information.
- Protect any user-provided text and stored data.
- Consult qualified mental health professionals for clinical concerns.

## 👨‍💻 Author

**Mohammed Salah Afzal**

- GitHub: [@MohammedSalahAfzal](https://github.com/MohammedSalahAfzal)
- LinkedIn: [Mohammed Salah Afzal](https://www.linkedin.com/in/mohammed-salah-afzal-27860031a/)
- Portfolio: [Personal Portfolio](https://incredible-speculoos-6282ce.netlify.app/)

## 📄 License

This project can be distributed under the MIT License if you choose to license it that way. Add a `LICENSE` file to the repository before claiming that the project is MIT-licensed.

---

⭐ If you find this project useful for learning NLP and deep learning, consider starring the repository.
