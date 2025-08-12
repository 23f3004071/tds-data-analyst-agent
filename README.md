# 📊 TDS Data Analyst Agent 

This project evaluates a **Data Analyst Agent API** using [Promptfoo](https://promptfoo.dev), an open-source framework for automated prompt and output testing.  
It is specifically designed to handle **multiple simultaneous requests**, with retry logic and strict timeouts, ensuring API compliance for competitive grading environments.

---

## 🚀 Features
- **Automated API Testing** with YAML test cases.
- **Multiple Question Handling** – one request per `questions.txt`.
- **Retry & Timeout Handling** – 4 retries with 5-minute limit each.
- **JSON Structure Validation** to match required output.
- **Custom Assertions** for numeric/text correctness.
- **LLM-based Rubric Grading** for visual outputs.

---

## 📂 Project Structure
- **`analyst.py`** – Main Python API file.
- **`eval.yaml`** – Promptfoo configuration & test cases.
- **`question.txt`** – questions files for each test.
- **`requirements.txt`** – Python dependencies.
- **`README.md`** – Project documentation.

---

## ⚙️ Installation
### 1️⃣ Clone the repository
```bash
git clone https://github.com/23f3004071/tds-data-analyst-agent.git
cd tds-data-analyst-agent
```
## 2️⃣ Install Python dependencies
```bash
pip install -r requirements.txt
```
## 3️⃣ Install Promptfoo 
```bash
npm install -g promptfoo
```
## 🔑 Environment Setup
Set your OpenAI API Key for LLM-based evaluation.

#### Windows (PowerShell)
```bash
setx OPENAI_API_KEY "your_api_key_here"
```
#### macOS / Linux
```bash
export OPENAI_API_KEY="your_api_key_here"
```

## ▶️ Running the API
Start the API locally:
```bash
python analyst.py
```
## 🧪 Running the Evaluation
Run Promptfoo with the provided config:
```bash
promptfoo eval -c eval.yaml
```