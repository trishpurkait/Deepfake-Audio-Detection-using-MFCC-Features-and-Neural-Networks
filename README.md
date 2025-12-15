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

### 1. Naive Bayes (GaussianNB)
Baseline machine learning model with moderate performance:

| Metric | Score |
|--------|-------|
| **Accuracy** | 67.12% |
| **Precision** | 77.55% |
| **Recall** | 47.30% |
| **F1 Score** | 58.77% |

### 2. Artificial Neural Network (ANN) ⭐
Deep learning model constructed using Keras/TensorFlow:

- **Architecture**: Sequential model with Dense layers (64 → 32 → 1)
- **Activation**: ReLU for hidden layers, Sigmoid for output layer
- **Training**: 20 Epochs, Batch Size of 16
- **Loss Function**: Binary Crossentropy

#### Performance Metrics:

| Metric | Score |
|--------|-------|
| **Accuracy** | **99.57%** |
| **Recall** | 99.25% |
| **F1 Score** | 97.91% |

## 🏆 Conclusion

The Artificial Neural Network (ANN) achieved outstanding performance with nearly **99.6% accuracy**. The extracted spectral features (MFCCs, Chroma, etc.) contain strong discriminative signals that the deep learning model successfully leverages to identify deepfake audio with high reliability.

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
