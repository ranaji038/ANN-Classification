Here is a clean and professional **README.md** for your Customer Churn Prediction Streamlit project.

---

# 📊 Customer Churn Prediction App

This project is a **Streamlit web application** that predicts whether a banking customer is likely to **churn** (close their account).
It uses a trained **Deep Learning model (TensorFlow/Keras)** along with **Label Encoders, One-Hot Encoders, and a Standard Scaler** to preprocess user inputs and generate predictions.

---

## 🚀 Features

- Modern, clean Streamlit UI
- Predicts customer churn probability
- Supports:

  - Gender encoding
  - Geography one-hot encoding
  - Scaling of numerical features

- Shows both **churn probability** and **final prediction**
- Easy-to-use, responsive layout

---

## 🧠 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **Streamlit**
- **scikit-learn**
- **Pandas / NumPy**
- **Pickle** (for serialized encoders)

---

## 🏗 Project Structure

```
│── model.h5                     # Trained deep learning model
│── label_encoder_gender.pkl     # LabelEncoder for gender
│── onehot_encoder_geo.pkl       # OneHotEncoder for geography
│── scaler.pkl                   # StandardScaler for preprocessing
│── app.py                       # Streamlit application file
│── README.md                    # Project documentation
```

---

## 🖥 How to Run the App Locally

### 1️⃣ Install Required Packages

```bash
pip install streamlit tensorflow scikit-learn pandas numpy
```

### 2️⃣ Place the Model & Encoder Files

Ensure the following files are present in the project folder:

- `model.h5`
- `label_encoder_gender.pkl`
- `onehot_encoder_geo.pkl`
- `scaler.pkl`

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

The app will open automatically in your browser at:

```
http://localhost:8501
```

---

## 🧩 How It Works

1. The app collects user inputs such as:

   - Geography
   - Gender
   - Age
   - Tenure
   - Balance
   - Credit Score
   - Number of Products
   - Active member status
   - Estimated Salary

2. Encoders/Scaler preprocess the input data to match model requirements.

3. The model outputs a **probability score** between `0.0` and `1.0`.

4. Based on the score:

   - If **> 0.5** → Customer is likely to churn
   - Otherwise → Customer is unlikely to churn

---

## 🖼 UI Preview

The UI includes:

- Labeled dropdowns and sliders
- Clean 3-column layout
- Metric display for final probability
- Bold success/error indicators for prediction

---

## 📄 License

This project is open-source under the **MIT License**.
