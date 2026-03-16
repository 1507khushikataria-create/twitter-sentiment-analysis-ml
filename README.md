# twitter-sentiment-analysis-ml
Twitter Sentiment Analysis using Machine Learning (TF-IDF + Logistic Regression) with scikit-learn.
# Twitter Sentiment Analysis using Machine Learning

## Project Overview

This project performs sentiment analysis on Twitter data using classical Natural Language Processing (NLP) techniques and machine learning models. The objective is to classify tweets into sentiment categories such as **Positive**, **Negative**, **Neutral**, and **Irrelevant**.

Sentiment analysis is an important NLP task used in various domains such as social media monitoring, customer feedback analysis, product reviews, and public opinion analysis.

This project demonstrates a complete **end-to-end NLP machine learning pipeline**, including:

* Data loading and inspection
* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Text vectorization using TF-IDF
* Model training using Logistic Regression
* Model evaluation using Accuracy and F1 Score
* Visualization of dataset distribution and results

The implementation uses Python and several widely used data science libraries including **Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn**.

---

# Dataset Description

The dataset used in this project consists of labeled tweets. Each record in the dataset contains information about the tweet topic, sentiment, and the tweet text itself.

### Dataset Columns

| Column    | Description                               |
| --------- | ----------------------------------------- |
| ID        | Unique identifier for each tweet          |
| Topic     | Topic related to the tweet                |
| Sentiment | Target label representing tweet sentiment |
| Tweet     | The actual tweet text                     |

### Sentiment Categories

The dataset includes four sentiment classes:

* Positive
* Negative
* Neutral
* Irrelevant

These labels represent the emotional tone or relevance of the tweet.

---

# Project Workflow

This project follows a standard machine learning pipeline used in many real-world NLP applications.

## 1. Data Loading

The dataset is loaded using Pandas and column names are assigned for clarity.

Key operations performed:

* Loading CSV file
* Assigning column names
* Inspecting dataset using `.head()` and `.info()`

---

## 2. Data Cleaning

Before training the model, the dataset is cleaned to remove incomplete entries.

Steps performed:

* Remove rows with missing tweet text
* Verify dataset structure
* Ensure text column is valid for processing

---

## 3. Exploratory Data Analysis (EDA)

EDA helps understand the dataset distribution before training the model.

### Sentiment Distribution

A count plot is generated to visualize how many tweets belong to each sentiment class.

This helps determine whether the dataset is balanced or imbalanced.

Example visualization included in the project:

![Sentiment Distribution](results/sentiment_distribution.png)

---

## 4. Feature and Target Selection

The dataset is divided into:

**Feature**

* Tweet text

**Target**

* Sentiment label

Feature:

```
Tweet text
```

Target:

```
Sentiment category
```

---

## 5. Text Vectorization

Machine learning models cannot process raw text directly. Therefore, tweet text is converted into numerical feature vectors.

This project uses **TF-IDF Vectorization**.

### TF-IDF (Term Frequency – Inverse Document Frequency)

TF-IDF converts text into numerical vectors based on:

* Word frequency in a tweet
* Importance of words across the entire dataset

This allows the model to identify important words related to sentiment.

Example transformation:

```
"I love this game" → [0.21, 0.00, 0.45, 0.67 ...]
```

---

## 6. Train-Test Split

The dataset is split into training and testing subsets using `train_test_split`.

Typical configuration:

* Training data: 80%
* Testing data: 20%

Purpose:

* Train the model on one portion
* Evaluate performance on unseen data

---

## 7. Model Training

The project uses **Logistic Regression**, a widely used machine learning algorithm for classification tasks.

Logistic Regression works well for text classification when combined with TF-IDF features.

Advantages:

* Fast training
* Good performance for high-dimensional data
* Common baseline model for NLP tasks

---

## 8. Model Evaluation

Model performance is evaluated using two metrics:

### Accuracy

Accuracy measures the percentage of correctly classified tweets.

Formula:

```
Accuracy = Correct Predictions / Total Predictions
```

---

### F1 Score

F1 Score balances precision and recall, making it useful for multi-class classification.

Weighted F1 Score is used because the dataset contains multiple sentiment classes.

---

### Confusion Matrix

A confusion matrix is used to visualize how well the model predicts each class.

Example visualization:

![Confusion Matrix](results/confusion_matrix.png)

---

# Technologies Used

The project was built using the following tools and libraries:

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

# Project Structure

```
twitter-sentiment-analysis
│
├── twitter_sentiment.py
├── README.md
│
└── results
     ├── sentiment_distribution.png
     └── confusion_matrix.png
```

---

# How to Run the Project

## Step 1 – Clone the repository

```
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
```

---

## Step 2 – Install dependencies

Install required Python libraries.

```
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Step 3 – Run the script

Execute the Python script.

```
python twitter_sentiment.py
```

The program will:

1. Load the dataset
2. Perform text vectorization
3. Train the model
4. Evaluate predictions
5. Display visualizations

---

# Results

The model achieves classification results based on the TF-IDF representation of tweet text combined with Logistic Regression.

Evaluation metrics include:

* Accuracy Score
* Weighted F1 Score
* Confusion Matrix visualization

---

# Future Improvements

Possible improvements for this project include:

* Implementing additional models such as Naive Bayes or Support Vector Machines
* Performing hyperparameter tuning
* Using deep learning models such as LSTM
* Applying transformer-based models like BERT
* Expanding the dataset for better generalization

---

# Learning Outcomes

Through this project, the following concepts were explored:

* Natural Language Processing basics
* Text vectorization techniques
* Machine learning classification workflows
* Model evaluation methods
* Data visualization for NLP tasks

---

# Author

Khushi Kataria
Machine Learning and AI enthusiast

---

# License

This project is intended for educational and learning purposes.

