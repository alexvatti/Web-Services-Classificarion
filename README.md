

# 🧠 Web Services Classification Using MAS and XAI

## 🚩 Problem Statement

The goal of this project is to classify web services into the **Top 30/40/50 categories** 
using machine learning models. However, we face several **challenges** that impact model performance:

- ✅ Data needs exploration, cleaning, and restructuring
- ⚠️ Severe **class imbalance** causes poor accuracy in minority classes
- 🧹 Pre-processing is essential for consistent input formatting
- 📊 Feature extraction (TF-IDF, SBERT) needs tuning for category distinction
- 🔍 Baseline model struggles to cross **80% accuracy**
- 📉 Model shows **overfitting** and fails to generalize well for all categories

---

## 🛠️ Plan of Action

1. **Data Exploration & Cleaning**
   - Merge and restructure multiple datasets
   - Handle missing values and normalize service descriptions

2. **Address Class Imbalance**
   - Use `SMOTE`, `class weights`, or `undersampling`
   - Visualize label distribution (Top 30, 40, 50)

3. **Preprocessing Pipeline**
   - Lowercase, remove stopwords, punctuation
   - Tokenization and lemmatization
   - Optional: BERT-style preprocessing for embedding methods

4. **Feature Extraction**
   - Combine `TF-IDF` + `SBERT embeddings`
   - Try `PCA` or `UMAP` for dimensionality reduction

5. **Baseline Model**
   - Models: `Logistic Regression`, `Naive Bayes`, `AdaBoost`
   - Train on balanced data subset to reach at least **80% accuracy**

6. **Model Evaluation**
   - Evaluate using `Precision`, `Recall`, `F1-score` across all classes
   - Use `Confusion Matrix` for error inspection

7. **Next Steps**
   - Fine-tune with `BERT`, or `XGBoost`
   - Implement explainability using `SHAP` / `LIME`

---


