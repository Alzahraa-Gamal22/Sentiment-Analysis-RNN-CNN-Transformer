# Sentiment Analysis: RNN vs CNN vs Transformer

## Project Overview
This project aims to compare three Deep Learning architectures — Recurrent Neural Networks (RNN), Convolutional Neural Networks (CNN), and Transformers — on the same sentiment analysis task using the IMDB Movie Reviews Dataset.

The goal is to evaluate their performance in classifying movie reviews as Positive or Negative and understand the differences between these architectures.

## Dataset
**IMDB Movie Reviews Dataset**

- Total original reviews: 50,000
- Positive reviews: 25,000
- Negative reviews: 25,000
- Task: Binary Sentiment Classification
- Input: Movie review text
- Output:
  - Positive (1)
  - Negative (0)

### Dataset After Cleaning
- Total reviews: 49,582
- Positive reviews: 24,884
- Negative reviews: 24,698
- Duplicate reviews removed: 418
- Missing values: 0

## Project Workflow

### 1. Data Exploration
- Loaded the IMDB dataset using Pandas.
- Inspected dataset structure and data types.
- Checked missing values.
- Identified duplicate reviews.
- Analyzed sentiment distribution.
- Inspected positive and negative review samples.
- Analyzed review lengths.

### 2. Data Preprocessing
- Removed duplicate reviews.
- Cleaned HTML tags and unnecessary whitespace.
- Converted sentiment labels into numerical values:
  - Positive = 1
  - Negative = 0
- Split the dataset into training, validation, and testing sets.
- Applied tokenization and sequence conversion.
- Applied padding to obtain fixed-length sequences.

### 3. Dataset Splitting

| Set | Percentage | Samples |
|---|---:|---:|
| Training | 70% | 34,707 |
| Validation | 15% | 7,437 |
| Testing | 15% | 7,438 |

Random state: 42

### 4. Model Development

Three architectures will be implemented and evaluated:

- **RNN:** Sequential processing of text data.
- **CNN (1D CNN):** Extracting local patterns and features from text sequences.
- **Transformer:** Using self-attention to capture relationships between words.

### 5. Model Evaluation

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training Time

### 6. Model Comparison

A final comparison will be conducted to analyze the performance of the three architectures and identify their strengths and weaknesses.

## Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers

## Project Roadmap

- [x] Dataset loading
- [x] Data exploration
- [x] Duplicate removal
- [x] Text cleaning
- [x] Label encoding
- [x] Train/Validation/Test splitting
- [x] Tokenization for RNN/CNN
- [x] Padding
- [ ] RNN implementation and training
- [ ] CNN implementation and training
- [ ]Transformer implementation and training
- [ ]Model evaluation
- [ ]Final comparison