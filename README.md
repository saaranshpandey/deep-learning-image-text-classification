# Deep Learning for Image & Text Classification
**CIS 519: Applied Machine Learning (Fall 2023)**  
Final Project – Image & Text Classification

---

## 1. Overview
This project examines two deep learning pipelines—one for **image classification** (CIFAR-10) and another for **text classification** (review sentiment)—and compares them against traditional ML approaches. Models explored include CNNs (with various regularization techniques, plus transfer learning) for images, and RNN-based methods (LSTM) or feed-forward networks (DAN) for text.

---

## 2. Project Files
- **`cis_project_cv.ipynb`**  
  - Implements CNN-based models for the CIFAR-10 dataset.  
  - Includes baseline CNN, improved CNN with dropout/L2/batch normalization, hyperparameter tuning, and dataset shift experiments (flip, zoom, rotation, brightness).  
  - Also explores transfer learning (VGG19, ResNet50) for improved accuracy.

- **`cis_project_nlp.ipynb`**  
  - Focuses on deep learning for text classification (review sentiment).  
  - Explores RNNs (LSTMs) with GloVe embeddings, plus DAN (Deep Averaging Network) models.  
  - Compares performance to a tuned logistic regression baseline.  
  - Investigates dataset shift scenarios (training on shorter reviews vs. testing on longer reviews).

---

## 3. Key Results

### Image Classification (CIFAR-10)
- **Baseline CNN** achieved ~71% accuracy.  
- **Improvements** with dropout, L2 regularization, and batch normalization raised accuracy to ~77–80%.  
- **Transfer Learning** (VGG19, ResNet50) further boosted accuracy to ~82–83%.  
- **Dataset Shifts** (flips, rotations, brightness changes) lowered performance, but deep models remained more robust than earlier traditional ML pipelines.

### Text Classification (Review Sentiment)
- **RNN (LSTM)** with GloVe embeddings reached ~88–89% accuracy on full-length reviews.  
- **DAN (w/ TF-IDF)** also showed strong performance, slightly behind the LSTM on test data.  
- **Comparison to Traditional ML** (optimized logistic regression) was close, but LSTM had better precision and overall robustness.  
- **Dataset Shift** (short vs. long reviews) reduced model accuracy, though deep models remained relatively resilient.

---

## 4. Conclusions
1. **CNNs** outperform traditional methods for image tasks, especially with transfer learning.  
2. **RNN-based models** (with suitable embeddings) match or surpass simpler ML models for text.  
3. **Data Shifts** degrade performance, but deep models remain more robust than traditional ML approaches.  
4. **Hyperparameter Tuning** (learning rates, layer dimensions, regularization, dropout) significantly enhances accuracy.

---
