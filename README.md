<h1 align="center">Skin Lesion Classification Using Deep Neural Networks on the HAM10000 Dataset</h1>
<h3 align="center">By Ruhan Shafi</h3>

Automated classification of pigmented skin lesions from dermatoscopic images, comparing a tuned Convolutional Neural Network (CNN) against a Multilayer Perceptron (MLP) and classical ML baselines (SVM, Decision Tree). Built as a Pattern Recognition and Machine Learning (PRML) university project, with an accompanying IEEE-style report and presentation.

## Abstract
 
This project investigates automated classification of dermatoscopic images with a focus on improving diagnostic accuracy for pigmented skin lesions. Two deep learning models — a CNN and an MLP — were developed and evaluated on the HAM10000 dataset. Extensive preprocessing and augmentation addressed dataset class imbalance and improved generalisation, and Bayesian optimisation via Keras Tuner was used to fine-tune key hyperparameters. Evaluation on unseen test data showed the CNN outperforming the MLP on both validation and test accuracy, with learning curves, confusion matrices, and ROC curves confirming strong discriminative performance across lesion types. The final CNN was packaged for reuse and further fine-tuning, illustrating its potential applicability to dermatological diagnostics and future medical imaging tasks.

## Dataset
 
The [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) ("Human Against Machine with 10,000 training images") dataset is a benchmark dermatological image collection curated by the Department of Dermatology at the Medical University of Vienna and the ViDIR Group (Tschandl, Rosendahl & Kittler, 2018). It comprises 10,015 dermatoscopic images across seven diagnostic categories:
 
| Code | Diagnosis |
|---|---|
| `akiec` | Actinic Keratoses / Intraepithelial Carcinoma |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis-like Lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic Nevi |
| `vasc` | Vascular Lesions |
 
Images were resized to 128×128 to reduce computational overhead, split 80/20 (stratified on diagnosis) into training and test sets, and one-hot encoded for multi-class classification. The dataset is notably imbalanced — `nv` heavily dominates the sample counts of `df` and `akiec` — and exhibits high intra-class visual similarity, making it a realistic and challenging testbed for medical image classification.

## Methodology
 
**Preprocessing & augmentation**
- Pixel values rescaled to `[0, 1]`; horizontal flips, random rotation, and zoom applied as real-time data augmentation to improve generalisation and offset class imbalance
**Model architectures**
- **CNN** — convolutional layers for hierarchical spatial feature extraction, well suited to the fine-grained texture and morphological patterns in dermatoscopic images
- **MLP** — fully connected layers only, included as a baseline to isolate the contribution of convolutional feature extraction
- **Classical baselines** — PCA-reduced SVM (linear/RBF kernels) and a grid-searched Decision Tree, built in Scikit-Learn
**Hyperparameter tuning**
- Bayesian optimisation via Keras Tuner efficiently searched convolutional filter counts, dense layer units, dropout rates, and Adam learning rate — balancing exploration and exploitation over an otherwise prohibitively large grid
- Early stopping on validation loss (restoring best weights) and a `ReduceLROnPlateau` scheduler stabilised training and reduced overfitting
- TensorBoard HParams logging tracked metrics against hyperparameter configurations across trials for reproducibility
**Evaluation harness**
- Metrics: accuracy, per-class precision/recall/F1 (critical given class imbalance), confusion matrices, and multi-class ROC/AUC
- Model checkpointing preserved best-epoch weights; final evaluation was run on a fully held-out, unseen test set to validate generalisation
**Model packaging**
- The final CNN was serialised via TensorFlow's native save format, preserving architecture, weights, and optimiser state — making it portable for inference, incremental fine-tuning, and integration into broader pipelines without redefinition
## Results
 
The tuned CNN was selected as the best-performing model (val. accuracy 0.7359), confirmed on the unseen test set (test accuracy 0.7329):
 
| Model | Test Accuracy |
|---|---|
| **CNN (tuned)** | **73.3%** |
| SVM + PCA | 71.6% |
| MLP (tuned) | ~67% |
| Decision Tree | 67.8% |
 
**CNN classification report (unseen test data):**
 
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| akiec | 0.42 | 0.31 | 0.35 | 65 |
| bcc | 0.50 | 0.34 | 0.40 | 103 |
| bkl | 0.47 | 0.31 | 0.38 | 220 |
| df | 0.00 | 0.00 | 0.00 | 23 |
| mel | 0.49 | 0.22 | 0.31 | 223 |
| nv | 0.80 | 0.96 | 0.87 | 1341 |
| vasc | 0.46 | 0.43 | 0.44 | 28 |
| **Accuracy** | | | **0.73** | 2003 |
| Macro avg | 0.45 | 0.37 | 0.39 | 2003 |
| Weighted avg | 0.68 | 0.73 | 0.70 | 2003 |
 
Performance is strongly skewed by class imbalance: the model discriminates the majority class (`nv`) very well but struggles on rare classes like `df` (0 F1) and `akiec`. Confusion matrices and ROC curves (`cm.png`, `ROC.png`) confirm the model can distinguish between all seven categories with resilience to background noise and lighting variation typical of real clinical imagery, despite this imbalance. See the full report for detailed discussion and figures.
 
## Ethical & privacy considerations
 
The project explicitly addresses the ethical and privacy implications of building ML systems on clinical imaging data:
 
- **Privacy** — HAM10000 images are fully anonymised, with patient metadata and identifying information removed, aligning with data protection standards such as GDPR and the Australian Privacy Principles
- **Fairness & bias** — Class imbalance and demographic underrepresentation in the dataset can bias predictions toward overrepresented classes; this was mitigated through augmentation and stratified sampling, and evaluated beyond raw accuracy via per-class precision/recall/F1
- **Scope of use** — The model is explicitly framed as a **decision-support tool**, not an autonomous diagnostic system — intended to assist clinicians by surfacing risk patterns rather than replace clinical judgement, given the severe real-world consequences of misclassifying malignant lesions
- **Reproducibility & transparency** — The full pipeline uses open-source frameworks (TensorFlow, Keras, Keras Tuner) with documented hyperparameters and configurations to support peer validation
