# Loan Eligibility Prediction Web App

## 📁 Description
This project is a full-stack web application that predicts a user's loan eligibility based on their financial and personal details. The app is built with a **React frontend**, **Flask backend**, and a **machine learning model** trained on historical loan data.


## 📁 Project Structure

```
loan-eligibility-app/
│
├── backend/           # Flask server handling API requests and ML predictions
├── frontend/          # React.js application for user interaction
├── model/             # Trained machine learning model (.pkl file)
├── train.csv          # Dataset used to train the model
├── README.md          # Project documentation
└── TrainAndSaveModel.ipynb  # Notebook to preprocess, train, and export the model
```


## 🔧 Backend

* Built with **Flask**
* Accepts POST requests at `/predict` with 11 user inputs
* Preprocesses data to match training pipeline
* Loads the trained model (`logistic_model.pkl`) and returns predictions
* Uses `flask-cors` to support requests from the React frontend

### Dependencies:

* Flask
* numpy
* pickle
* flask-cors


## 💻 Frontend

* Built with **React.js** (Vite or Create React App)
* Accepts user input through form fields
* Sends data to backend `/predict` route via `fetch`
* Displays model prediction as "Approved" or "Rejected"
* Styled using **Tailwind CSS** with subtle animations


## 🧠 Model

* Trained using **RandomForestClassifier**
* Uses 11 engineered features like Gender, Income, Credit History, etc.
* Trained and saved using `TrainAndSaveModel.ipynb`
* Model is serialized using `pickle` and stored at `model/logistic_model.pkl`

## 📈 Algorithms Used

1. **RandomForestClassifier**

   * An ensemble learning method that builds multiple decision trees and merges their results to improve accuracy and reduce overfitting.

2. **LogisticRegression** (Optional)

   * A linear classification algorithm used for binary outcomes (like "Approved" vs "Rejected").

> You can switch between algorithms in the training notebook by changing the classifier line.

## ⚡ How to Run the Project

### Prerequisites:

* Python 3.x
* Node.js & npm
* pip (Python package installer)

### 1. Clone the Repository

```
git clone <your-repo-url>
cd loan-eligibility-app
```

### 2. Backend Setup

```
cd backend
pip install -r requirements.txt
python app.py
```

### 3. Frontend Setup

```
cd frontend
npm install
npm run dev
```

### 4. Train the Model (Optional if model is not already saved)

```
Open TrainAndSaveModel.ipynb in Jupyter or VS Code
Run all cells to save the model to /model/logistic_model.pkl
```

> Make sure your Flask backend is running at `http://localhost:5000` when testing the frontend.

## ✅ Result

* Provides instant loan eligibility predictions
* Successfully integrates ML with a responsive UI
* Handles edge cases and invalid inputs with graceful messages


## 📚 Learning

* Hands-on practice integrating **ML + Flask + React**
* How to preprocess and encode categorical data
* Deploying full-stack ML apps locally
* Managing API-CORS issues and model versioning

## 🚀 Future Improvements

* Deploy to cloud (Render, Vercel, Railway, etc.)
* Add login/authentication support
* Save prediction history to a database (e.g. SQLite or MongoDB)
* Add more advanced ML models or tuning (GridSearchCV)
* Improve UI with charts or personalized feedback

---

> Built with ❤️ using Python, React, and Machine Learning
