# Deepfake Audio Detection using MFCC Features and Neural Networks

A machine learning project that detects deepfake audio with **99.57% accuracy** using Mel-Frequency Cepstral Coefficients (MFCC) features and deep neural networks.

## 📌 Project Overview

This project builds a robust system for detecting synthetic (deepfake) audio by analyzing audio files and extracting key acoustic features. The system utilizes both machine learning and deep learning classifiers to distinguish between "Real" and "Fake" audio recordings.

## 🛠️ Workflow & Methodology

### 1. Data Preparation
- **Source**: Audio files categorized into `real` and `fake`, extracted from zip archives hosted on Google Drive
- **Sampling**: Audio processed at 16,000 Hz sampling rate
- **Dataset Size**: 22,859 audio samples

### 2. Feature Extraction
Built using the `librosa` library to extract **48 numerical features** from each audio clip:

- **MFCCs**: 20 Mel-frequency cepstral coefficients (mean values)
- **Spectral Centroid**: Center of mass of the spectrum
- **Spectral Bandwidth**: Width of the frequency band
- **Spectral Rolloff**: Frequency threshold for spectral energy
- **Spectral Contrast**: Amplitude difference between peaks and valleys
- **Chroma**: Pitch class profiles
- **Tonnetz**: Tonal space features
- **Zero Crossing Rate**: Rate of sign-changes in the signal
- **RMS**: Root Mean Square energy

### 3. Data Preprocessing
- **Duplicate Check**: Verified no duplicate rows exist
- **Train-Test Split**: 80% Training, 20% Testing
- **Scaling**: Features normalized using both `StandardScaler` and `MinMaxScaler`

### 4. Exploratory Data Analysis (EDA)
- Distribution analysis using histograms and boxplots
- Feature correlation analysis with target labels
- Class balance verification (Label 1 = Real, Label 0 = Fake)

## 🧠 Modeling & Results

Multiple machine learning and deep learning algorithms were trained and evaluated on the extracted audio features:

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|---------|----------|
| **Support Vector Machine (SVM)** ⭐ | **99.56%** | **99.47%** | **99.65%** | **99.56%** |
| K-Nearest Neighbors (KNN) | 98.75% | 97.83% | 99.69% | 98.75% |
| Random Forest | 98.05% | 98.01% | 98.06% | 98.03% |
| Artificial Neural Network (ANN) | 97.90% | 96.60% | 99.25% | 97.91% |
| Decision Tree | 92.48% | 91.85% | 93.07% | 92.45% |
| Naive Bayes | 67.13% | 77.55% | 47.31% | 58.77% |

### Detailed Model Descriptions

#### 1. Support Vector Machine (SVM) ⭐ **Best Performer**
- **Algorithm**: Support Vector Classifier with default parameters
- **Performance**: Achieved the highest overall accuracy and balanced metrics
- **Confusion Matrix**: [[2296, 12], [8, 2256]]
- **Strengths**: Excellent precision-recall balance, minimal false positives/negatives

#### 2. K-Nearest Neighbors (KNN)
- **Algorithm**: KNN classifier with default k value
- **Performance**: Second-best performer with highest recall
- **Confusion Matrix**: [[2258, 50], [7, 2257]]
- **Strengths**: Exceptional recall (99.69%), very few false negatives

#### 3. Random Forest
- **Algorithm**: Ensemble of decision trees
- **Performance**: Strong and consistent results across all metrics
- **Confusion Matrix**: [[2263, 45], [44, 2220]]
- **Strengths**: Robust performance with good generalization

#### 4. Artificial Neural Network (ANN)
- **Architecture**: Sequential model with Dense layers (64 → 32 → 1)
- **Activation**: ReLU for hidden layers, Sigmoid for output layer
- **Training**: 20 Epochs, Batch Size of 16, Binary Crossentropy loss
- **Confusion Matrix**: [[2229, 79], [17, 2247]]
- **Strengths**: High recall (99.25%), good at detecting fake audio

#### 5. Decision Tree
- **Algorithm**: Single decision tree classifier
- **Performance**: Moderate performance, prone to overfitting
- **Confusion Matrix**: [[2121, 187], [157, 2107]]

#### 6. Naive Bayes
- **Algorithm**: Gaussian Naive Bayes classifier
- **Performance**: Baseline model with lowest accuracy
- **Confusion Matrix**: [[1998, 310], [1193, 1071]]
- **Limitations**: Low recall (47.31%), misses many positive samples

## 🏆 Conclusion

The **Support Vector Machine (SVM)** emerged as the best-performing model with **99.56% accuracy**, demonstrating exceptional precision and recall balance. The extracted spectral features (MFCCs, Chroma, etc.) contain strong discriminative signals that enable multiple algorithms to achieve over 97% accuracy. Traditional ML algorithms (SVM, KNN, Random Forest) outperformed the deep learning approach (ANN) in this specific task, likely due to the relatively small feature space (48 features) where linear and distance-based methods excel.

## 📦 Dependencies

```bash
pip install librosa numpy pandas scikit-learn tensorflow matplotlib seaborn
```

**Required Libraries:**
- Python 3.x
- Librosa
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/Keras
- Matplotlib & Seaborn

## 📂 Project Structure

```
├── DeepFake_Audio_Detection.ipynb    # Main implementation notebook
├── extracted_audio_features.csv      # Extracted MFCC features (48 features × 22,859 samples)
└── README.md                          # Project documentation
```

## 🚀 Usage

1. Clone the repository:
```bash
git clone https://github.com/trishpurkait/Deepfake-Audio-Detection-using-MFCC-Features-and-Neural-Networks.git
cd Deepfake-Audio-Detection-using-MFCC-Features-and-Neural-Networks
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open and run the Jupyter notebook:
```bash
jupyter notebook DeepFake_Audio_Detection.ipynb
```

## 📊 Dataset

This project uses the [SceneFake dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/scenefake) from Kaggle, containing both real and deepfake audio samples.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📧 Contact

**Trish Purkait**
- GitHub: [@trishpurkait](https://github.com/trishpurkait)
- Email: trishpurkait@gmail.com

## 🙏 Acknowledgments

- Dataset: [Mohammed Abdeldayem - SceneFake Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/scenefake)
- Feature extraction powered by librosa library
