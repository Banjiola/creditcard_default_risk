# Credit Card Default PredicFraud Detection
 

> *Using machine learning to accurately predict credit card defaulters and maximise profits by accurately predicting credit card non defaulters.*

---

## 📌 Table of Contents
- [Overview](##overview)
- [Business Objective](#business-objective)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Results & Insights](#results--insights)
- [Author](#author)

---

## 🧩 Overview
<p style="font-size: 14px;text-align: justify;">
Credit card default risk modelling has significant real-world applications for financial institutions, as the approval of credit card loans for consumers carries the risk of financial losses due to non-payment or late repayment. However, it also results in profits if more consumers repay their loans. The major challenge for these institutions is to award as many loans as possible to non-defaulters and fewer loans to defaulters. Consequently, the goal of this classification task depends on the institution's risk tolerance. While risk-averse institutions focus on minimising false negatives (classifying defaulters as non-defaulters), growth-oriented institutions may prioritise minimising false positives (classifying non-defaulters as defaulters).
<p>

---

## 🎯 Business Objective


> To help financial institutions minimise financial losses due to credit card default .

---

## 📊 Dataset

- **Source**: [UCI ML Repo](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients )
- **Size**: e.g., 30,000 rows × 23 columns


---

## 🔎 Exploratory Data Analysis
## Results for EDA (See reports/EDA for all plots)
---

<p style="font-size: 14px;text-align: justify;">
An EDA of the dataset revealed that consumers with a university degree accounted for 47% of the data by education level, while single individuals accounted for 53% of consumers by marital status, with 60% of females comprising the consumer base, as illustrated in Figures 1, 2, and 3 respectively.
</p>


  <figure>
    <img src="reports\EDA\education.png" width=600>
    <figcaption style="font-size: 16px;"><b>Figure 1. Distribution of Consumers by Education</b></figcaption>
  </figure>


  <figure>
    <img src="reports\EDA\marital_status.png" width=600>
    <figcaption style="font-size: 16px;"><b> Figure 2. Distribution of Consumers by Marital Status</b></figcaption>
  </figure>

  <figure>
    <img src="reports\EDA\Number of loan applicants by Marital and Default Status.png" width=600>
    <figcaption style="font-size: 16px;"><b> Figure 3. Distribution of Consumers by Marital and Default Status</b></figcaption>
  </figure>



---
## 🏗️ Feature Engineering

- Creation of Payment Consistency Feature
- Creation of Payment Delays Feature


---

## 📈 Results & Insights

Table 1. ROC-AUC and Precision Recall AUC (All Models)
| Model                   | ROC-AUC | Precision-Recall AUC |
|-------------------------|---------|---------------------|
| Decision Tree           | 0.73    | 0.49                |
| Logistic Regression     | 0.72    | 0.48                |
| XGBoost                 | 0.77    | 0.526               |
| K Nearest Neighbour     | 0.76    | 0.525               |
| Support Vector Machines | 0.75    | 0.49                |



Table 2. Performance on Test Set (Top Three Models Models)
| Model                   | Macro F1 | Macro Recall |
|-------------------------|----------|--------------|
| XGBoost                 | **0.72** | **0.73**     |
| Support Vector Machines | 0.71     | **0.73**     |
| K Nearest Neighbour     | 0.69     | 0.71         |


---

## 👨‍💻 Author

**Olabanji Olaniyan**  

Data Scientist  
📫 [LinkedIn](https://www.linkedin.com/in/olabanji-olaniyan-59a6b0198/) | [Portfolio](banjiola.github.io/Olabanji-Olaniyan/)
