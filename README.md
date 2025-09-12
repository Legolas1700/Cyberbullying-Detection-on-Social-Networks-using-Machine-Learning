# 🛡️ Cyberbullying Detection App

A Streamlit web application that detects cyberbullying in social media text, aiming to create a safer and more positive online environment.  
The app classifies input text into **Cyberbullying** or **Non-Cyberbullying** with confidence scores and suggests moderation actions.

---

## 🚀 Features
- Real-time cyberbullying detection from user input
- Confidence percentage for each prediction
- Suggested moderation actions (ban user, restrict comments, no action)
- Visualization of prediction results with interactive charts
- Dataset overview and insights (class distribution, top reasons)

---

## 🛠 Tech Stack
- **Python 3.11+**
- **Streamlit** → Web app framework  
- **scikit-learn** → Machine learning (Naive Bayes, TF-IDF)  
- **NLTK** → Natural language preprocessing (stopwords, lemmatization)  
- **Pandas & NumPy** → Data manipulation  
- **Matplotlib, Seaborn, Plotly** → Data visualization  

---

## ⚙️ Setup Instructions

### 1. Download Project
Save all files in a single folder (e.g. `Cyberbullying-Detection-App`):
```

Cyberbullying-Detection-App/
├── app.py
├── cyberbullying\_detection.py
├── cyberbullying\_dataset.csv
├── requirements.txt
├── README.md
└── LICENSE

````

### 2. Create Virtual Environment (Recommended)
**Windows**
```bash
python -m venv venv
venv\Scripts\activate
````

**Mac/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

Inside the project folder:

```bash
streamlit run app.py
```

### 5. Close the App Safely

Press:

```
CTRL + C
```

in the terminal where the app is running.

### 6. Run Again Later

* Activate your virtual environment
* Run:

```bash
streamlit run app.py
```

---

## 📂 Dataset

* Default dataset included: **`cyberbullying_dataset.csv`**
* You can also upload your own CSV (must contain a `Text` column and labels for classification).

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## ✨ Acknowledgements

* [NLTK](https://www.nltk.org/)
* [scikit-learn](https://scikit-learn.org/)
* [Streamlit](https://streamlit.io/)


