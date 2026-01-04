# 🛡️ Adaptive Fraud Detection Engine
### Production-Style Real-Time ML Decisioning System

## 📌 Overview
This project implements a **production-style fraud detection system** designed to simulate how fraud detection operates in real FinTech and banking environments.

Rather than focusing only on model accuracy, the system emphasises:
- Real-time fraud decisioning
- Cost-sensitive risk trade-offs
- Analyst review and case management workflows
- Operational monitoring and transparency

The goal is to bridge the gap between **machine learning models** and **real-world fraud operations**.

---

## 🎯 Key Objectives
- Detect fraudulent transactions in near real time  
- Optimise decisions based on **business cost**, not accuracy alone  
- Support analyst workflows through case management  
- Provide explainability and operational visibility  
- Deploy the system as a containerised web application  

---

## 🧠 Machine Learning & Decision Strategy
- **Model:** Gradient Boosted Trees (optimised for highly imbalanced tabular data)  
- **Input:** Anonymised transaction features + transaction amount  
- **Output:** Fraud probability score per transaction  

### Decision Logic
Instead of a fixed threshold, the system uses **cost-sensitive thresholding**:
- False Negatives are treated as significantly more expensive than False Positives  
- Each transaction is classified into a **business decision**:
  - **APPROVE**
  - **REVIEW**

This mirrors real-world fraud risk policies used by financial institutions.

---

## 🧩 Core System Components

### 1️⃣ Live Transaction Feed
- Simulated real-time transaction stream  
- Instant fraud scoring per transaction  
- Visual risk indicators for fast triage  

### 2️⃣ Batch Scoring
- CSV-based ingestion for bulk transaction scoring  
- Designed for backfills and daily extracts  
- Outputs fraud probability and business decision  

### 3️⃣ Single Transaction Analysis
- Manual feature input for ad-hoc investigation  
- SHAP explainability:
  - Top risk-increasing features  
  - Top risk-reducing features  
- Improves analyst trust and model transparency  

### 4️⃣ Case Management & Auto-Resolution (Demo)
- REVIEW decisions automatically create cases  
- Tracks:
  - Case status  
  - Resolution source  
  - SLAs and timestamps  
- Auto-resolution logic simulates customer confirmation and back-office actions  

---

## 📊 Operational Analytics (Ops View)
A dedicated operational layer monitors:
- Transactions per minute  
- Review rate and case backlog  
- False positive rate  
- Case resolution rate  
- Average time to resolution  
- Risk concentration by hour and day  

This layer is critical for understanding **system health and workload**, not just model performance.

---

## 🛠️ Tech Stack
- **Programming:** Python  
- **Machine Learning:** Gradient Boosted Trees  
- **Explainability:** SHAP  
- **Web App:** Streamlit  
- **Containerisation:** Docker  
- **Cloud Deployment:** AWS EC2  

---

## 🌐 Live Demo
- **Live Application:** [AWS deployment link]  
- **Technical Report:** [PDF link]  

---

## 🚀 Why This Project Matters
Most fraud detection projects stop at *"the model works"*.

This project focuses on **production-level questions**:
- How do fraud decisions scale operationally?
- How do analysts interact with ML systems?
- How is financial risk measured beyond AUC?
- What does a deployable fraud system actually look like?

---

## 🔮 Future Enhancements
- Adaptive thresholding based on fraud-rate drift  
- Feedback loops from resolved cases  
- Active learning for analyst-reviewed transactions  
- Event-driven architecture for true real-time ingestion  

---

## 👤 Author
**Vachan Sardar**  
Data Scientist | ML Engineer  
Focused on FinTech, Risk Analytics, and Production ML Systems
