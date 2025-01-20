# RAG Chatbot

RAG Chatbot is an  chatbot application built using **React** for the frontend, **Tailwind CSS** for styling, and **FastAPI** for the backend. The backend leverages **LLM (Large Language Models)** and **RAG (Retrieval-Augmented Generation)** for intelligent and context-aware responses.

---

## Features
- Modern UI built with React and Tailwind CSS.
- Backend powered by FastAPI for efficient and scalable API management.
- Uses Retrieval-Augmented Generation (RAG) for enhanced question-answering capabilities.
- Integration with Large Language Models (LLM) to generate responses.
- Persistent chat storage using session-based storage.

---

## Prerequisites
Before setting up the project, ensure you have the following installed:

- **Node.js** (v16 or later)
- **Python** (v3.8 or later)
- **npm** or **yarn**
- **pip** (Python package installer)

---

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/hkinnovapath/RAG_Chatbot.git
cd RAG_Chatbot
```

---

## UI Setup

1. Navigate to the frontend directory:
   ```bash
   cd ./ai_chatbot_ui
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm start
   ```

The UI will be accessible at `http://localhost:3000`.

---

## Backend Setup

### Step 1: Navigate to Backend Directory
```bash
cd ./api_with_llm
```

### Step 2: Create a Python Virtual Environment
```bash
python -m venv myenv
```

### Step 3: Activate the Virtual Environment
- **Windows**:
  ```bash
  cd ./myenv/Scripts/
  ./activate
  ```

- **Mac/Linux**:
  ```bash
  source myenv/bin/activate
  ```

### Step 4: Install Required Packages
```bash
pip install -r ./requirements.txt
```

### Step 5: Start the Backend Server
```bash
uvicorn app.main:app --reload
```

The FastAPI server will be accessible at `http://127.0.0.1:8000`.

---

## Project Structure

```
RAG_Chatbot/
|
|-- ai_chatbot_ui/           # Frontend (React + Tailwind)
|   |-- src/                # React Components
|   |-- public/             # Static Assets
|   |-- other Config files  # other Config files
|   |-- package.json        # Frontend Dependencies
|
|-- api_with_llm/           # Backend (FastAPI)
|   |-- app/                # FastAPI Application
|   |   |-- main.py         # Entry Point for Backend
|   |   |-- other files/    # other files
|   |-- requirements.txt    # Python Dependencies
|
|-- README.md               # Project Documentation
```

---

## Tech Stack

### Frontend
- **React**: JavaScript library for building user interfaces.
- **Tailwind CSS**: Utility-first CSS framework for styling.

### Backend
- **FastAPI**: Modern web framework for Python.
- **RAG**: Retrieval-Augmented Generation for answering questions.
- **LLM**: Integration with Large Language Models for intelligent responses.

---

## Usage

1. Start the **backend server** using FastAPI.
2. Run the **frontend** React application.
3. Access the application at `http://localhost:3000` and interact with the chatbot.

---

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

---

## Acknowledgments

Project made with ❤️ by **Hemant** and **Jafar**.