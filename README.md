🛡 AI-Based Fake Internship & Job Detection System
📌 Overview

The AI-Based Fake Internship & Job Detection System is a Machine Learning web application that detects whether a job or internship posting is Real or Fake.

With the rise of online job scams, many students and job seekers fall victim to fraudulent offers that demand registration fees or promise unrealistic salaries.
This system uses Natural Language Processing (NLP) and Logistic Regression to analyze job postings and provide a fraud risk prediction.

🚀 Features

Detects Fake or Real job postings

Displays Fraud Risk Score (%)

Considers Company Name + Job Description

Machine Learning-based prediction

Interactive modern UI

Visual REAL / FAKE stamp output



🧠 How It Works

User enters a job description.

The system combines:

Job Title

Company Name

Job Description

Text is converted into numerical format using TF-IDF Vectorizer.

A Logistic Regression model predicts:

0 → Real Job

1 → Fake Job

Fraud probability is shown as a Risk Score (%).



🛠 Technologies Used

Python

Pandas

Scikit-learn

Flask

HTML & CSS

Logistic Regression

TF-IDF Vectorization
