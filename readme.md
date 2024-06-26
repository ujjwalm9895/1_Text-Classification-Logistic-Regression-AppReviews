# Text Processing and Classification Using Logistic Regression

### Business Overview

Natural Language Processing (NLP) is the field that enables machines to understand, read, and gain insights from human language. NLP involves the automatic handling of human languages and is flourishing thanks to the availability of vast data and computational power. NLP has applications in healthcare, media, finance, human resources, and more. It continues to grow with each passing day.

---

### Aim

The primary objective of this project is to understand basic text preprocessing and build a classification model using logistic regression.

---

### Data Description

The dataset contains over a thousand reviews about an application that is openly available to the public. The data includes reviews, sentiments (positive or negative), and various other variables.

---

### Tech Stack

- Language: `Python`
- Libraries: `pandas`, `seaborn`, `matplotlib`, `sklearn`, `nltk`

---

## Approach

1. Data Description and Visualization
2. Introduction to the NLTK library
3. Data Preprocessing:
   a. Conversion to lowercase
   b. Tokenization
   c. Stopwords removal
   d. Punctuation removal
   e. Stemming

4. Bag of Words:
   a. Binary
   b. Non-binary
   c. N-grams
5. TF-IDF
6. Model Building and Accuracy
7. Predictions on New Reviews

---

## Modular Code Overview

1. **Input**: Contains the data used for analysis, including:
   - `Canva_reviews.xlsx`

2. **Source**: Contains modularized code for all the project steps, including:
   - `model.py`
   - `processing.py`
   - `utils.py`

   These Python files contain functions used in the `Engine.py` file.

3. **Output**: Contains pre-trained models and vectorizers required for the project, including:
   - [List of pre-trained models and vectorizers]

4. **config.py**: Contains project configurations.

5. **Engine.py**: The main file to run the entire project, which trains the model and saves it in the output folder.

---
