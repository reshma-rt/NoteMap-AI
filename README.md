# <p align="center">🗺️ NoteMap — AI-Powered Study Tool</p>

<p align="center">
<img src="https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white" />
<img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" />
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" />
</p>

<p align="center">
<strong>NoteMap</strong> is a serverless, AI-powered study platform that converts raw learning materials into structured notes and interactive flashcards.  
Using <strong>Amazon Bedrock</strong>, OCR pipelines, and intelligent parsing, it transforms PDFs and documents into organized study assets for efficient learning.
</p>

---

# 📸 Project Walkthrough

## 🏠 Home Page
<p align="center">
<img src="./Assets/Home%20Page.jpeg" width="800"/>
</p>

*Landing page introducing NoteMap with authentication access.*

---

## 📤 Upload Page
<p align="center">
<img src="./Assets/Upload%20Page.jpeg" width="800"/>
</p>

*Drag-and-drop interface allowing users to upload study materials for AI processing.*

---

## 📊 Dashboard Page
<p align="center">
<img src="./Assets/Dashboard%20Page.jpeg" width="800"/>
</p>

*Central workspace displaying structured notes, topics, and study progress.*

---

## 🧠 Interactive Flashcards
<p align="center">
<img src="./Assets/FlashCard.jpeg" width="800"/>
</p>

*AI-generated flashcards designed for active recall and quick revision.*

---

# ⚡ Key Features

### 🤖 Intelligent OCR Processing
Hybrid OCR pipeline combining:

- `pdf.js` for PDF text extraction  
- `Tesseract.js` for browser-based OCR  
- Python backend using **OpenCV preprocessing** (deskewing, denoising)

---

### 🧠 AI Content Analysis
- Powered by **Amazon Bedrock**
- Converts extracted text into structured **JSON knowledge graphs**
- Identifies topics, headings, and key concepts automatically

---

### 🗂️ Automated Flashcards
- Extracts important concepts
- Generates **Q&A flashcards**
- Helps with **active recall learning**

---

### 📊 Responsive Dashboard
- Dark / Light mode
- Organized knowledge view
- Topic filters and quick search

---

### 📥 Dynamic Export
Export structured notes into:

- 📄 PDF  
- 📝 TXT  
- 📑 Word  

Generated using **jsPDF**

---

### 🔐 Secure Cloud Infrastructure
- **Authentication:** AWS Cognito  
- **Storage:** Amazon S3  
- **Database:** DynamoDB  
- **Compute:** AWS Lambda + API Gateway  

---

# 🛠️ Tech Stack

## Frontend

- **React.js**
- **Tailwind CSS**
- **Lucide React Icons**
- **AWS Amplify**
- **pdf.js**
- **Tesseract.js**
- **jsPDF**

---

## Backend & AI Pipeline

- **AWS Lambda**
- **Amazon Bedrock**
- **Amazon DynamoDB**
- **Amazon S3**
- **Python (Flask)**

Libraries used:

- spaCy  
- NLTK  
- KeyBERT  
- OpenCV  

---

# 🚀 How It Works

1️⃣ User uploads study material (PDF/Image)  
2️⃣ OCR pipeline extracts raw text  
3️⃣ AI (Amazon Bedrock) analyzes the content  
4️⃣ Concepts and sections are structured into notes  
5️⃣ Flashcards are generated automatically  
6️⃣ Users review content on the dashboard
