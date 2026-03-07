📘 NoteMap AI — Smart Study Notes Organizer

AI-powered application that converts PDFs and handwritten notes into structured study material, automatically generating organized notes, sections, and flashcards for better learning.

🚀 Overview

NoteMap AI is an intelligent study assistant that helps students convert raw study material into structured, searchable, and downloadable notes.

Users can upload PDFs or images of notes, and the system performs OCR-based text extraction, organizes the content into sections, generates flashcards, and stores the notes securely in the cloud.

The project demonstrates integration of AI, OCR, cloud services, and full-stack development.

✨ Key Features
📄 Smart File Processing

Upload PDF, PNG, JPG documents

Extract text using OCR (Tesseract.js)

Convert scanned notes into readable text

🧠 Automatic Note Organization

Detect sections and structure notes

Generate summarized content blocks

Automatically categorize notes by subject

🎓 Flashcard Generation

Create flashcards from extracted content

Interactive flashcard study mode

Shuffle and navigate between flashcards

☁️ Cloud Storage

Store files securely in AWS S3

Save structured notes in DynamoDB

User authentication via AWS Cognito

📊 Smart Dashboard

View all processed notes

Filter notes by subject

Search through stored notes

📥 Export Notes

Download notes in multiple formats:

TXT

PDF

Word Document

🏗 System Architecture
User Upload (PDF / Image)
        │
        ▼
React Frontend
        │
        ▼
OCR Processing (Tesseract.js)
        │
        ▼
Text Processing & Structuring
        │
        ▼
AWS Backend
 ├── API Gateway
 ├── AWS Lambda
 ├── DynamoDB
 └── S3 Storage
        │
        ▼
User Dashboard + Flashcards
🛠 Tech Stack
Frontend

React.js

JavaScript

Tesseract.js (OCR)

PDF.js

jsPDF

Tailwind UI Components

Lucide Icons

Backend

Python

Flask API

Cloud Services (AWS)

AWS Cognito (Authentication)

AWS API Gateway

AWS Lambda

AWS DynamoDB

AWS S3 Storage

AWS Amplify

Libraries

OpenCV

Pillow

NumPy

pdf2image

NLTK

spaCy

KeyBERT

YAKE

Sentence Transformers

📂 Project Structure
NoteMap-AI
│
├── frontend
│   ├── App.js
│   ├── components
│   └── UI logic
│
├── backend
│   ├── app.py
│   ├── OCR pipeline
│   └── NLP processing
│
├── cloud
│   ├── AWS Lambda
│   ├── DynamoDB integration
│   └── S3 storage logic
│
└── README.md
⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/yourusername/notemap-ai.git
cd notemap-ai
2️⃣ Install Frontend Dependencies
npm install
3️⃣ Install Backend Dependencies
pip install -r requirements.txt
4️⃣ Run Backend
python app.py

Backend runs at:

http://localhost:5000
5️⃣ Run Frontend
npm start

Frontend runs at:

http://localhost:3000
🔐 Authentication

The application uses AWS Cognito for secure authentication.

Features:

Email login

Google OAuth login

Secure token-based API requests
