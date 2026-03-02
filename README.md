# Sign Language MNIST — Translation Engine

A translation engine that takes a static image of a hand sign (A–Z) and predicts the corresponding letter. Built with a classical ML baseline (Random Forest) and a multi-layer perceptron (TensorFlow), with a Streamlit dashboard for live inference.

## Dataset

[Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) — 28×28 grayscale images of hand signs for letters A–Z (excluding J and Z, which require motion). The dataset contains ~27,455 training and ~7,172 test samples across 24 classes.

## Project Structure

```
├── baseline_model/
│   ├── model.ipynb              # Baseline training notebook
│   └── model.py                 # Standalone baseline training script
├── multilayer_model/
│   ├── multi_layer.py           # MLP architecture definition
│   ├── train_multi_layer.py     # Training script
│   ├── test_multi_layer.py      # Evaluation script
│   ├── multi_layer.ipynb        # MLP notebook (Colab)
│   └── multilayer_model.keras   # Saved trained model
├── utils/
│   └── data_process.py          # Data loading utilities
├── data/
│   └── sign-language-mnist/     # Dataset CSVs and reference images
├── dashboard.py                 # Streamlit app for inference
├── requirements.txt
└── README.md
```

## Models

### Baseline — Random Forest

- Uses `RandomForestClassifier` with `RandomizedSearchCV` for hyperparameter tuning
- Features are standardized with `StandardScaler`
- Saved as `baseline_model.pkl`

### Multilayer — MLP (TensorFlow/Keras)

- Sequential model: Dense(512) → Dense(256) → Dense(128) → Dense(25)
- Each hidden layer uses BatchNormalization, LeakyReLU, Dropout(0.25), and L2 regularization
- Adam optimizer with exponential learning rate decay
- EarlyStopping and ReduceLROnPlateau callbacks
- Saved as `multilayer_model.keras`

## Setup

### 1. Download the data

Download the Sign Language MNIST dataset and place it in `data/sign-language-mnist/`. Requires a [Kaggle API](https://github.com/Kaggle/kaggle-api) key in `~/.kaggle/kaggle.json`.

```bash
kaggle datasets download -d datamunge/sign-language-mnist
unzip sign-language-mnist.zip -d data/sign-language-mnist/
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate.bat     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Train the baseline model

Run the notebook `baseline_model/model.ipynb` or the standalone script:

```bash
python baseline_model/model.py
```

### Train the multilayer model

```bash
python multilayer_model/train_multi_layer.py
```

### Evaluate the multilayer model

```bash
python multilayer_model/test_multi_layer.py
```

### Run the dashboard

```bash
streamlit run dashboard.py
```

Upload a PNG/JPG image of a hand sign and the dashboard will display the predicted letter and confidence score. The dashboard uses the trained MLP model for inference.
