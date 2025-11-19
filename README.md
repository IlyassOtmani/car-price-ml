
*# 🚗 Car Price Prediction with CI/CD

[![CI](https://github.com/IlyassOtmani/car-price-ml/workflows/Continuous%20Integration/badge.svg)](https://github.com/IlyassOtmani/car-price-ml/actions)
[![CD](https://github.com/IlyassOtmani/car-price-ml/workflows/Continuous%20Deployment/badge.svg)](https://github.com/IlyassOtmani/car-price-ml/actions)

An end-to-end machine learning project for predicting car prices with automated CI/CD pipelines using GitHub Actions. Built for automotive market analysis and price estimation.

## 📊 Dataset

*Car Price Prediction Dataset* from [Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)

- *205 car samples* with detailed specifications
- *26 features* including:
  - *Engine*: size, type, horsepower, cylinders, fuel system
  - *Dimensions*: wheelbase, length, width, height, weight
  - *Performance*: city/highway MPG, peak RPM, compression ratio
  - *Configuration*: body type, drive wheel, fuel type, aspiration
  - *Brand*: extracted from car name
  - *Risk Rating*: insurance symboling

## 🎯 Project Overview

This project implements a *Random Forest Regression* model to predict car prices with:
- ✅ Automated model training and evaluation
- ✅ Performance metrics and visualization
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Automatic deployment to Hugging Face Spaces
- ✅ Interactive Gradio web interface
- ✅ Feature engineering (brand extraction, data cleaning)

## 🏗️ Architecture


├── Data/
│   └── CarPrice_Assignment.csv  # Car price dataset
├── Model/
│   └── model_pipeline.skops     # Trained model (auto-generated)
├── Results/
│   ├── metrics.txt              # Model metrics (auto-generated)
│   └── model_results.png        # Visualizations (auto-generated)
├── App/
│   ├── app.py                   # Gradio interface
│   └── requirements.txt         # App dependencies
├── .github/workflows/
│   ├── ci.yml                   # Continuous Integration
│   └── cd.yml                   # Continuous Deployment
├── train.py                     # Training script
├── Makefile                     # Automation commands
└── requirements.txt             # Project dependencies


## 🚀 Quick Start

### 1. Clone the Repository
bash
git clone https://github.com/IlyassOtmani/car-price-ml.git
cd car-price-ml


### 2. Install Dependencies
bash
make install


### 3. Download Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction) and place CarPrice_Assignment.csv in the Data/ folder.

### 4. Train the Model
bash
make train


### 5. Run the Web App Locally
bash
cd App
python app.py


## 🤖 CI/CD Pipeline

### Continuous Integration (CI)
Triggered on every push to main branch:
1. ✅ Install dependencies
2. ✅ Format code with Black
3. ✅ Train the model
4. ✅ Generate performance report with CML
5. ✅ Commit results back to repository

### Continuous Deployment (CD)
Triggered after successful CI:
1. ✅ Deploy to Hugging Face Spaces
2. ✅ Make the app publicly accessible

## 📈 Model Performance

Performance metrics are automatically updated in Results/metrics.txt after each training run. Typical results:
- *RMSE*: ~$2,000 - $3,500
- *MAE*: ~$1,500 - $2,500  
- *R² Score*: ~0.85 - 0.92

The model explains approximately 85-92% of the variance in car prices.

## 🛠️ Technologies Used

- *ML Framework*: scikit-learn
- *Model Persistence*: skops
- *Web Interface*: Gradio
- *CI/CD*: GitHub Actions
- *MLOps*: CML (Continuous Machine Learning)
- *Deployment*: Hugging Face Spaces
- *Code Formatting*: Black

## 🔧 Configuration

### GitHub Secrets
Set up the following secrets in your GitHub repository:

1. *For CI:*
   - USER_NAME: IlyassOtmani
   - USER_EMAIL: ilyassotmani555@gmail.com

2. *For CD:*
   - HF_TOKEN:

### Hugging Face Deployment
Update the deployment command in Makefile with your Hugging Face username:
makefile
deploy:
    pip install huggingface-hub
    python -c "from huggingface_hub import HfApi; \
HfApi().create_repo(repo_id='IlyassOtmani/car-price-predictor', ...)"


## 📝 Usage

### Training
bash
# Full pipeline
make all

# Individual steps
make install    # Install dependencies
make format     # Format code
make train      # Train model
make eval       # Generate evaluation report


### Running the App
bash
cd App
python app.py


Then open your browser to http://localhost:7860

### Making Predictions

*Example 1: Economy Car (Toyota Corolla-like)*
- Brand: Toyota
- Body: Sedan, 4 doors
- Engine: 130 cu.in, 102 HP, 4 cylinders
- MPG: 24 city / 30 highway
- Weight: 2,385 lbs
- *Predicted*: ~$7,000 - $9,000

*Example 2: Luxury Sedan (BMW-like)*
- Brand: BMW
- Body: Sedan, 4 doors
- Engine: 209 cu.in, 182 HP, 6 cylinders
- MPG: 16 city / 22 highway
- Weight: 3,740 lbs
- *Predicted*: ~$35,000 - $45,000

*Example 3: Sports Car (Porsche-like)*
- Brand: Porsche
- Body: Hardtop, 2 doors
- Engine: 194 cu.in, 207 HP, 6 cylinders, Turbo
- MPG: 17 city / 25 highway
- Weight: 2,778 lbs
- *Predicted*: ~$40,000 - $50,000

## 📊 Feature Importance

The most important features for car price prediction:
1. *Engine Size* - Larger engines = higher price
2. *Curb Weight* - Heavier cars typically more expensive
3. *Horsepower* - Performance correlates with price
4. *Car Brand* - Luxury brands command premium
5. *Body Type* - Convertibles and hardtops cost more
6. *Engine Type* - DOHC and advanced engines increase value
7. *City/Highway MPG* - Fuel efficiency affects pricing

## 🧪 Model Details

### Preprocessing
- *Brand Extraction*: Extracts brand from car name and fixes misspellings
- *Numerical Features*: Median imputation + Standard scaling
- *Categorical Features*: One-hot encoding with unknown handling

### Algorithm
- *Model*: Random Forest Regressor
- *Estimators*: 100 trees
- *Random State*: 125 (for reproducibility)
- *Parallel Processing*: n_jobs=-1

### Train/Test Split
- *Training*: 70% (~143 samples)
- *Testing*: 30% (~62 samples)

## 🔍 Data Preprocessing

The training script includes:
1. *Car Brand Extraction*: Splits car name to extract brand
2. *Spelling Corrections*: Fixes common brand misspellings (vw→volkswagen, toyouta→toyota, etc.)
3. *Missing Value Handling*: Median imputation for numerical, constant for categorical
4. *Feature Scaling*: StandardScaler for numerical features
5. *Encoding*: One-hot encoding for categorical features

## 📈 Visualizations

The model generates four comprehensive plots:
1. *Predicted vs Actual*: Scatter plot showing prediction accuracy
2. *Residual Plot*: Shows prediction errors across price ranges
3. *Distribution Comparison*: Histogram comparing actual vs predicted distributions
4. *Error Distribution*: Shows the distribution of absolute errors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

## 📊 Dataset Characteristics

- *Target Variable*: Price (continuous)
- *Categorical Features*: 10 (fuel type, aspiration, body type, brand, etc.)
- *Numerical Features*: 14 (dimensions, performance metrics, weight, etc.)
- *Total Samples*: 205
- *No Missing Values*: After preprocessing

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline development
- Feature engineering and data preprocessing
- Model training and evaluation
- CI/CD for machine learning
- Model deployment and serving
- Interactive web application development
- MLOps best practices

## 🙏 Acknowledgments

- Dataset provided by Hellbuoy on Kaggle
- Based on automotive pricing analysis for market entry
- Built with scikit-learn, Gradio, and GitHub Actions

