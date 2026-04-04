🦠 ARPModel: Antibiotic Resistance Prediction

📌 Overview

Antibiotic resistance is a growing global health challenge. ARPModel is a machine learning–powered web app that predicts bacterial resistance patterns based on numeric antibiotic sensitivity values.
Built with Python, scikit‑learn, and Streamlit, the app provides real‑time predictions, decision support insights, and interactive visualizations to help clinicians and researchers interpret resistance data.

🚀 Features

- ML Prediction: Trained on numeric antibiotic sensitivity values (not binary thresholds).
- Decision Support Tool: Rule‑based insights (thresholds, ranges, risk scoring) layered on top of predictions.
- Interactive UI: Clean Streamlit dashboard with inputs on the left and outputs on the right.
- Visualization: Gene network graphs and resistance patterns for deeper analysis.
- Deployment: Hosted on Streamlit Cloud, accessible via any browser.

🛠️ Tech Stack

- Python
- scikit‑learn (ML model training)
- pandas (data handling)
- matplotlib / seaborn (visualizations)
- networkx (gene network visualization)
- Streamlit (UI + deployment)

📂 Project Structure


├── app.py                 # Main Streamlit app

├── train.py               # Model training script

├── model.pkl              # Saved ML model

├── label_encoder.pkl      # Saved label encoder

├── dataset.csv            # Antibiotic resistance dataset

├── requirements.txt       # Dependencies

└── README.md              # Project documentation




⚙️ Installation & Usage
1. Clone the Repository
   
   git clone https://github.com/ronaksharma2908/arpmodel.git
   cd arpmodel


2. Install Dependencies
   
   pip install -r requirements.txt


3. Train the Model (if needed)
   
   python train.py


4. Run the App
   
   streamlit run app.py



🌐 Deployment

  The app is deployed on Streamlit Cloud.

  👉 Live Demo Link (https://arpmodel-ronaksharma29.streamlit.app/)


📊 Example Workflow

- Enter numeric antibiotic values (e.g., Imipenem = 25, Ceftazidime = 30).
- Model predicts resistant/sensitive location.
- Decision Support panel highlights thresholds, ranges, and risk scores.
- Visualizations show gene networks and resistance trends.

🏆 Hackathon Context

This project was developed for a Hackathon on Antibiotic Resistance Prediction.
It demonstrates how ML + decision support + visualization can create a real‑time, judge‑ready clinical tool.

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

📜 License

This project is licensed under the MIT License
