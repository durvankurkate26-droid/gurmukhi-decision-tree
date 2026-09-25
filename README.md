# Decision Tree-based Handwritten Gurmukhi Character Recognition

## Overview
This project implements a handwritten Gurmukhi character recognition system using a Decision Tree classifier. It covers dataset exploration, image preprocessing, HOG feature extraction, model training and hyperparameter tuning, evaluation, and an overfitting/misclassification analysis.

## Dataset
- Source: [Gurmukhi Dataset – Mendeley Data](https://data.mendeley.com/datasets/h65gdk4ptv/1)
- 41 character classes, 12,128 total images
- Balanced dataset (~291–299 images per class)
- Original image size varies per sample (resized during preprocessing)

## Pipeline
1. **Preprocessing**: Grayscale conversion → Otsu thresholding (binarization) → Resize to 64×64 → Normalize to [0,1]
2. **Feature Extraction**: Histogram of Oriented Gradients (HOG), producing a 1764-dimensional feature vector per image
3. **Model**: `DecisionTreeClassifier` (scikit-learn), tuned via `GridSearchCV` (3-fold CV, 40 parameter combinations)
4. **Train/Test Split**: 80/20, stratified by class (no feature scaling applied — Decision Trees split on raw thresholds and don't require normalized input)

## Results
- **Best parameters**: `criterion='entropy'`, `max_depth=20`, `min_samples_leaf=1`
- **Test Accuracy: 30.59%**
- Macro F1-score: 0.31
- Train accuracy: 100.00% (indicates significant overfitting)
- Strongest classes: 9 (F1 = 0.60), 36 (0.52), 24 & 30 (0.50 each), 8 (0.50)
- Weakest classes: 21 (F1 = 0.13), 32 (0.17), 1 & 6 (0.19 each)

## Overfitting Analysis
Diagnostics revealed a persistent train/test gap regardless of tuning:
- A depth-capped tree (max_depth=20) already overfit: 68.93% train vs 23.21% test
- Removing the depth cap entirely pushed training accuracy to 100.00%, but test accuracy barely moved (24.36%)
- GridSearchCV across 40 combinations of `max_depth`, `min_samples_leaf`, and split criterion selected the best-performing combination, improving test accuracy to 30.59% — but the train/test gap (100% vs ~31%) persisted

This points to a structural limitation of a single Decision Tree on this feature representation: with 1764 correlated HOG features and 41 classes, axis-aligned single-feature splits struggle to carve stable, generalizable decision regions — a fundamentally different approach from a margin-based classifier that combines information across all features simultaneously.

## Comparison with Support Vector Machine (SVM)
An SVM (RBF kernel, C=10, gamma='scale') trained on the identical 1764-dimensional HOG feature space achieves a substantially higher test accuracy (high 80s%) on the same dataset. The Decision Tree trains far faster (22.62 seconds vs the SVM's much longer training time) and needs no feature scaling, but its axis-aligned splitting cannot model the smooth, high-dimensional boundaries needed to separate 41 visually similar handwritten characters as effectively as a kernel-based method can.

## Tech Stack
- Python, OpenCV, scikit-image, scikit-learn, NumPy, Matplotlib, Seaborn
- Developed and trained on Google Colab

## Reproducing
1. Download the dataset from the Mendeley link above
2. Run `CIS_VCG.ipynb` end to end (mount Google Drive, update the dataset path to match your own Drive structure)
3. Preprocessed images and extracted HOG features are cached as `.npy` files on Drive for faster re-runs across sessions
