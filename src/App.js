import React, { useState, useRef, useEffect } from 'react';
import { Home, Upload, LayoutDashboard, Sun, Moon, FileText, Search, Download, ChevronDown, ChevronRight, X, CheckCircle, Clock, AlertCircle, Eye, BookOpen, Tag, MapPin, Sparkles, Brain, Zap, ChevronLeft, Shuffle } from 'lucide-react';
import jsPDF from 'jspdf';
import * as pdfjsLib from "pdfjs-dist";
pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.js';
export default function NoteMap() {
  const [currentPage, setCurrentPage] = useState('home');
  const [isDark, setIsDark] = useState(true);
  const [selectedSubject, setSelectedSubject] = useState('All Subjects');
  const [searchQuery, setSearchQuery] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [dragActive, setDragActive] = useState(false)
  const [previewFile, setPreviewFile] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const [processedNotes, setProcessedNotes] = useState([]);
  const [apiEndpoint, setApiEndpoint] = useState('https://j84ctsm6jj.execute-api.us-east-1.amazonaws.com/default/NoteMap-AI-Processor');
  const [backendInfo, setBackendInfo] = useState(null);
  const [expandedTOC, setExpandedTOC] = useState({});
  const [selectedFormat, setSelectedFormat] = useState('pdf');
  const [downloadStatus, setDownloadStatus] = useState({});
  const [tocSearchQuery, setTocSearchQuery] = useState('');
  const [selectedTOCEntry, setSelectedTOCEntry] = useState(null);
  const [showFlashcards, setShowFlashcards] = useState(false);
  const [currentFlashcardIndex, setCurrentFlashcardIndex] = useState(0);
  const [isFlashcardFlipped, setIsFlashcardFlipped] = useState(false);
  const [flashcards, setFlashcards] = useState([]);
  const [showPDFPreview, setShowPDFPreview] = useState(false);
  const [pdfPreviewUrl, setPDFPreviewUrl] = useState(null);
  const [showViewDashboardButton, setShowViewDashboardButton] = useState(false);
  const fileInputRef = useRef(null);
 const [showFlashcardModal, setShowFlashcardModal] = useState(false);
const [activeNoteForFlashcards, setActiveNoteForFlashcards] = useState(null);
const [isFlipped, setIsFlipped] = useState(false);
const [loading, setLoading] = useState(false);

  const cleanExtractedText = (text) => {
    if (!text) return '';
    return text
      .replace(/ð·/g, '•')
      .replace(/â€¢/g, '•')
      .replace(/â€œ|â€/g, '"')
      .replace(/â€˜|â€™/g, "'")
      .replace(/â€"|â€"/g, '-')
      .replace(/([A-Za-z])\s+(?=[A-Za-z])/g, '$1')
      .replace(/\s+/g, ' ')
      .trim();
  };

  const SUBJECTS = [
    'Mathematics',
    'Physics',
    'Chemistry',
    'Biology',
    'Computer Science',
    'Data Science & Artificial Intelligence',
    'Engineering (General)',
    'Environmental Science',
    'Economics',
    'Psychology',
    'Sociology',
    'Political Science',
    'History',
    'Literature & Languages',
    'Business & Management',
    'Other'
  ];

  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const res = await fetch('http://localhost:5000/api/health');
      const data = await res.json();
      setBackendInfo(data);
    } catch (err) {
      console.error('Backend not available:', err);
    }
  };

const handleFileUpload = async (files) => {
  // 1. Validation Logic
  const validFiles = Array.from(files).filter(file => {
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'application/pdf'];
    const maxSize = 10 * 1024 * 1024;

    if (!validTypes.includes(file.type)) {
      alert(`${file.name} is not a valid file type.`);
      return false;
    }
    if (file.size > maxSize) {
      alert(`${file.name} is too large. Max 10MB.`);
      return false;
    }
    return true;
  });

  if (validFiles.length === 0) return;

  // 2. Set Loading States
  setIsProcessing(true);
  setLoading(true);
  setShowViewDashboardButton(false);

  // Use Promise.all to handle multiple files concurrently
  const processingPromises = validFiles.map(async (file) => {
    
    // Duplicate Check
    const isDuplicate = uploadedFiles.some(f => f.name === file.name) || 
                        processedNotes.some(n => n.filename === file.name);
    
    if (isDuplicate) {
      console.warn(`File ${file.name} is already uploaded.`);
      return; 
    }

    // 3. Process each file
    return new Promise((resolve) => {
      const reader = new FileReader();
      
      reader.onload = async (e) => {
        const fileId = Date.now() + Math.random();
        const newFile = {
          id: fileId,
          name: file.name,
          size: (file.size / 1024).toFixed(2) + ' KB',
          type: file.type,
          status: 'processing',
          progress: 0,
          previewUrl: e.target.result,
          file: file
        };

        setUploadedFiles(prev => [...prev, newFile]);

        if (file.type === 'application/pdf') {
          setPDFPreviewUrl(e.target.result);
          setShowPDFPreview(true);
        }

        // Progress Animation
        let progress = 0;
        const progressInterval = setInterval(() => {
          progress += 15;
          if (progress <= 90) {
            setUploadedFiles(prev => prev.map(f =>
              f.id === fileId ? { ...f, progress } : f
            ));
          }
        }, 400);

        try {
          const result = await extractTextFromFile(file);
          clearInterval(progressInterval);

          if (!result || !result.success) {
            throw new Error(result?.error || 'Processing failed');
          }

          // 4. Map Data - UPDATED TO MATCH NEW LAMBDA STRUCTURE
          const processedNote = {
            id: result.id || fileId,
            filename: file?.name || 'Untitled_Document',
            
            // NEW: Subject from Lambda
            subject: result.subject || 'General Study',
            
            // Date
            date: result.date || new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
            
            // Statistics
            charCount: result.charCount || 0,
            wordCount: result.wordCount || 0,
            
            // Text Content
            extractedText: result.text || '',
            preview: (result.text || '').substring(0, 150) + '...',
            
            // NEW: Sections array with {title, desc} structure
            sections: result.sections || [],
            
            // NEW: Concepts array
            concepts: result.concepts || [],
            
            // Flashcards (generated from sections and concepts)
            flashcards: [],
            
            // UI Display
            totalSections: (result.sections || []).length,
            
            // Organized notes for display (using NEW structure)
            organizedNotes: result.sections && result.sections.length > 0 
              ? result.sections.map(s => `### ${s.title}\n${s.desc}`).join('\n\n')
              : result.text || "Processing complete...", 
              
            // Metadata
            timestamp: new Date().toLocaleTimeString()
          };

          // Generate Flashcards from sections and concepts
          processedNote.flashcards = generateFlashcards(processedNote);
          
          // 5. Update All States
          setProcessedNotes(prev => [...prev, processedNote]);
          setFlashcards(prev => [{
            id: processedNote.id,
            fileName: file.name,
            content: result.text,
            organizedNotes: processedNote.organizedNotes,
            timestamp: processedNote.timestamp
          }, ...prev]);

          setUploadedFiles(prev => prev.map(f => 
            f.id === fileId ? { ...f, status: 'completed', progress: 100 } : f
          ));
          
          setShowViewDashboardButton(true);
          resolve();

        } catch (error) {
          clearInterval(progressInterval);
          console.error('Processing error:', error);
          setUploadedFiles(prev => prev.map(f =>
            f.id === fileId ? { ...f, status: 'error', progress: 0, errorMessage: error.message } : f
          ));
          alert(`Failed to process ${file.name}: ${error.message}`);
          resolve();
        }
      };

      reader.readAsDataURL(file);
    });
  });

  // 6. Cleanup after all files are done
  await Promise.all(processingPromises);
  setIsProcessing(false);
  setLoading(false);
};


// App.js (React)

const extractTextFromFile = async (file) => {
  const awsEndpoint = "https://j84ctsm6jj.execute-api.us-east-1.amazonaws.com/default/NoteMap-AI-Processor";
  const imagesToProcess = [];

  // 1. Convert PDF or Image into Base64 strings for the AWS Lambda
  if (file.type === "application/pdf") {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const viewport = page.getViewport({ scale: 1.5 });
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      canvas.width = viewport.width;
      canvas.height = viewport.height;

      await page.render({ canvasContext: context, viewport }).promise;
      imagesToProcess.push(canvas.toDataURL("image/jpeg", 0.7).split(",")[1]);
    }
  } else {
    const base64 = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.readAsDataURL(file);
    });
    imagesToProcess.push(base64);
  }

  // 2. Process all pages/images through AWS and collect the full results
  const allText = [];
  const allSections = [];
  const allConcepts = [];
  let totalWords = 0;
  let totalChars = 0;
  let detectedSubject = "Other";

  for (const imgB64 of imagesToProcess) {
    try {
      const res = await fetch(awsEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imgB64 }),
      });

      if (!res.ok) throw new Error(`AWS error ${res.status}`);

      const data = await res.json();

      // NEW LAMBDA STRUCTURE: data now has sections with {title, desc}
      if (data.text) {
        allText.push(data.text);
        totalChars += data.charCount || data.text.length;
        totalWords += data.wordCount || data.text.split(/\s+/).length;
      }
      
      // Capture sections array directly from Lambda
      if (data.sections && Array.isArray(data.sections)) {
        allSections.push(...data.sections);
      }
      
      // Capture concepts array
      if (data.concepts && Array.isArray(data.concepts)) {
        allConcepts.push(...data.concepts);
      }
      
      // Update subject if detected
      if (data.subject && data.subject !== "General") {
        detectedSubject = data.subject;
      }

    } catch (err) {
      console.error("Error processing page through AWS:", err);
    }
  }

  // 3. Return the complete object matching the new Lambda structure
  return {
    success: true,
    text: allText.join("\n\n"),
    
    // Sections with title and desc (not content/details)
    sections: allSections.length > 0 ? allSections : [
      { 
        title: "Note Summary", 
        desc: allText.join(" ").substring(0, 500) + "..." 
      }
    ],
    
    // Clean up duplicate concepts
    concepts: [...new Set(allConcepts)],
    
    // Subject classification
    subject: detectedSubject,
    
    // Statistics
    wordCount: totalWords,
    charCount: totalChars,
    
    // Date
    date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  };
};

  const handlePreview = (file) => {
    setPreviewFile(file);
    setShowPreview(true);
  };

  const closePreview = () => {
    setShowPreview(false);
    setPreviewFile(null);
  };

  const downloadNote = async (format, note) => {
    setDownloadStatus(prev => ({ ...prev, [note.id]: 'Preparing download...' }));
    setTimeout(() => {
      let content = '';
      let filename = '';
      let mimeType = '';

      if (format === 'txt') {
        content = generateTextContent(note);
        filename = `${note.filename.replace(/\.[^/.]+$/, '')}_notes.txt`;
        mimeType = 'text/plain';

        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');

        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        setDownloadStatus(prev => ({ ...prev, [note.id]: `Downloaded ${filename}` }));

      } else if (format === 'pdf') {
        filename = `${note.filename.replace(/\.[^/.]+$/, '')}_notes.pdf`;

        try {
          const doc = new jsPDF();
          const pageWidth = doc.internal.pageSize.getWidth();
          const pageHeight = doc.internal.pageSize.getHeight();
          const margin = 20;
          const maxWidth = pageWidth - (margin * 2);
          let yPosition = 20;

          // Helper function to clean text of problematic characters
          const cleanText = (text) => {
            if (!text) return '';
            return text
              .replace(/[^\x00-\x7F]/g, '') // Remove non-ASCII characters
              .replace(/\s+/g, ' ') // Normalize whitespace
              .trim();
          };

          // Helper function to add text with page breaks
          const addText = (text, fontSize, isBold = false, color = [0, 0, 0]) => {
            doc.setFontSize(fontSize);
            doc.setFont(undefined, isBold ? 'bold' : 'normal');
            doc.setTextColor(...color);

            const cleanedText = cleanText(text);
            const lines = doc.splitTextToSize(cleanedText, maxWidth);
            lines.forEach(line => {
              if (yPosition > pageHeight - 30) {
                doc.addPage();
                yPosition = 20;
              }
              doc.text(line, margin, yPosition);
              yPosition += fontSize * 0.5;
            });

            return yPosition;
          };

          // Add title
          const topicColor = getSubjectRGBColor(note.subject);
          addText(note.filename, 20, true, topicColor);
          yPosition += 5;

          // Add a colored line under title
          doc.setDrawColor(...topicColor);
          doc.setLineWidth(1);
          doc.line(margin, yPosition, pageWidth - margin, yPosition);
          yPosition += 10;

          // Add metadata
          addText(`Subject: ${note.subject} | Date: ${note.date} | Sections: ${note.totalSections}`, 10, false, [100, 100, 100]);
          yPosition += 10;

          // Add "Content" heading
          addText('Content', 16, true, topicColor);
          yPosition += 5;

          // Add sections
          note.sections.forEach((section, idx) => {
            yPosition += 5;

            // Section title
            addText(`${idx + 1}. ${section.title}`, 14, true, [55, 65, 81]);
            yPosition += 3;

            // Section description
            addText(section.desc, 11, false, [75, 85, 99]);
            yPosition += 8;
          });

          // Save the PDF
          doc.save(filename);

          setDownloadStatus(prev => ({ ...prev, [note.id]: `Downloaded ${filename}` }));
        } catch (error) {
          console.error('PDF generation error:', error);
          alert('Failed to generate PDF. Please try TXT or DOCX format.');
          setDownloadStatus(prev => ({ ...prev, [note.id]: `Error generating PDF` }));
        }

      } else if (format === 'docx') {
        content = generateWordContent(note);
        filename = `${note.filename.replace(/\.[^/.]+$/, '')}_notes.doc`;
        mimeType = 'application/msword';

        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        setDownloadStatus(prev => ({ ...prev, [note.id]: `Downloaded as Word document` }));
      }

      setTimeout(() => {
        setDownloadStatus(prev => {
          const newStatus = { ...prev };
          delete newStatus[note.id];
          return newStatus;
        });
      }, 3000);
    }, 500);
  };

  const generateTextContent = (note) => {
    // Helper function to clean text
    const cleanText = (text) => {
      if (!text) return '';
      return text
        .replace(/[^\x00-\x7F]/g, '') // Remove non-ASCII characters
        .replace(/\s+/g, ' ') // Normalize whitespace
        .trim();
    };

    let content = [];
    content.push('='.repeat(80));
    content.push('ORGANIZED NOTES - NoteMap');
    content.push(`Generated: ${new Date().toLocaleString()}`);
    content.push('='.repeat(80));
    content.push('');
    content.push(`FILE: ${cleanText(note.filename)}`);
    content.push(`SUBJECT: ${cleanText(note.subject)}`);
    content.push(`DATE: ${cleanText(note.date)}`);
    content.push(`SECTIONS: ${note.totalSections}`);
    content.push('='.repeat(80));
    content.push('');

    content.push('CONTENT:');
    content.push('-'.repeat(80));
    note.sections.forEach((section, idx) => {
      content.push(`\n${idx + 1}. ${cleanText(section.title)}`);
      if (section.topic) {
        content.push(`   Topic: ${cleanText(section.topic)}`);
      }
      if (section.page) {
        content.push(`   Page: ${section.page} | Level: H${section.level}`);
      }
      content.push(`   ${cleanText(section.desc)}`);
      content.push('');
    });
    if (note.concepts && note.concepts.length > 0) {
      content.push('KEY CONCEPTS:');
      content.push('-'.repeat(80));
      content.push(note.concepts.map(c => cleanText(c)).join(', '));
      content.push('');
    }

    return content.join('\n');
  };

  const generateWordContent = (note) => {
    const topicColor = getSubjectHexColor(note.subject);
    // Helper function to clean text
    const cleanText = (text) => {
      if (!text) return '';
      return text
        .replace(/[^\x00-\x7F]/g, '') // Remove non-ASCII
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
        .replace(/\n/g, '<br>')
        .trim();
    };

    let html = `<!DOCTYPE html>
<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
<head>
  <meta charset='utf-8'>
  <title>${cleanText(note.filename)} - Organized Notes</title>
  <style>
    body {
      font-family: Calibri, Arial, sans-serif;
      line-height: 1.6;
      color: #1f2937;
      margin: 40px;
    }
    h1 {
      color: ${topicColor};
      font-size: 28pt;
      margin-bottom: 10px;
      border-bottom: 3px solid ${topicColor};
      padding-bottom: 10px;
    }
    h2 {
      color: ${topicColor};
      font-size: 20pt;
      margin-top: 24px;
      margin-bottom: 12px;
      border-bottom: 2px solid #e5e7eb;
      padding-bottom: 8px;
    }
    h3 {
      color: #374151;
      font-size: 16pt;
      margin-top: 16px;
      margin-bottom: 8px;
    }
    p {
      color: #4b5563;
      font-size: 11pt;
      margin-bottom: 12px;
    }
    .metadata {
      color: #6b7280;
      font-size: 10pt;
      margin-bottom: 20px;
    }
    .badge {
      background-color: ${topicColor};
      color: white;
      padding: 4px 12px;
      border-radius: 12px;
      font-weight: bold;
      font-size: 9pt;
      display: inline-block;
      margin-right: 10px;
    }
  </style>
</head>
<body>
  <h1>${cleanText(note.filename)}</h1>
  <div class="metadata">
    <span class="badge">${cleanText(note.subject)}</span>
    <span>Date: ${cleanText(note.date)}</span> |
    <span>Sections: ${note.totalSections}</span> | 
    <span>Words: ${note.wordCount || 0}</span>
  </div>
  <h2>Content</h2>
`;
    note.sections.forEach((section, idx) => {
      html += `  <h3>${idx + 1}. ${cleanText(section.title)}</h3>\n`;
      html += `  <p>${cleanText(section.desc)}</p>\n`;
    });
    html += `</body></html>`;
    return html;
  };

  // Helper function to escape HTML special characters and preserve formatting
  const escapeHtml = (text) => {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML
      .replace(/\n/g, '<br>')
      .replace(/\t/g, '&nbsp;&nbsp;&nbsp;&nbsp;');
  };

  const generatePDFContent = (note) => {
    const topicColor = getSubjectHexColor(note.subject);
    let html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <title>${escapeHtml(note.filename)} - Organized Notes</title>
  <style>
    @media print {
      body { margin: 0; }
      .page-break { page-break-before: always; }
    }
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      line-height: 1.6;
      color: #1f2937;
      max-width: 210mm;
      margin: 0 auto;
      padding: 20mm;
      background: white;
    }
    .header {
      border-bottom: 3px solid ${topicColor};
      padding-bottom: 20px;
      margin-bottom: 30px;
    }
    h1 {
      color: ${topicColor};
      font-size: 32px;
      margin: 0 0 10px 0;
    }
    h2 {
      color: ${topicColor};
      font-size: 24px;
      margin-top: 30px;
      margin-bottom: 15px;
      border-bottom: 2px solid #e5e7eb;
      padding-bottom: 10px;
    }
    h3 {
      color: #374151;
      font-size: 18px;
      margin-top: 20px;
      margin-bottom: 10px;
    }
    p {
      color: #4b5563;
      font-size: 14px;
      margin-bottom: 15px;
      text-align: justify;
    }
    .metadata {
      color: #6b7280;
      font-size: 12px;
      margin-bottom: 20px;
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>${escapeHtml(note.filename)}</h1>
    <div class="metadata">
      <strong>Subject:</strong> ${escapeHtml(note.subject)} |
      <strong>Date:</strong> ${escapeHtml(note.date)} | 
      <strong>Sections:</strong> ${note.totalSections}
    </div>
  </div>
  <h2>Content</h2>
`;
    note.sections.forEach((section, idx) => {
      html += `  <h3>${idx + 1}. ${escapeHtml(section.title)}</h3>\n`;
      html += `  <p>${escapeHtml(section.desc)}</p>\n`;
    });
    html += `</body></html>`;
    return html;
  };

  const getSubjectHexColor = (subject) => {
    const colors = {
      'Mathematics': '#6366f1',
      'Physics': '#3b82f6',
      'Chemistry': '#10b981',
      'Biology': '#059669',
      'Computer Science': '#8b5cf6',
      'Other': '#6b7280'
    };
    return colors[subject] || colors['Other'];
  };

  const getSubjectRGBColor = (subject) => {
    const colors = {
      'Mathematics': [99, 102, 241], // Indigo
      'Physics': [59, 130, 246], // Blue
      'Chemistry': [16, 185, 129], // Green
      'Biology': [5, 150, 105], // Emerald
      'Computer Science': [139, 92, 246], // Purple
      'Data Science & Artificial Intelligence': [168, 85, 247], // Purple
      'Engineering (General)': [245, 158, 11], // Amber
      'Environmental Science': [34, 197, 94], // Green
      'Economics': [234, 179, 8], // Yellow
      'Psychology': [236, 72, 153], // Pink
      'Sociology': [251, 146, 60], // Orange
      'Political Science': [239, 68, 68], // Red
      'History': [161, 98, 7], // Brown
      'Literature & Languages': [14, 165, 233], // Sky blue
      'Business & Management': [20, 184, 166], // Teal
      'Other': [107, 114, 128] // Gray
    };
    return colors[subject] || colors['Other'];
  };

  const getTopicColor = (topic) => {
    const colors = {
      'Mathematics': 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30',
      'Physics': 'bg-blue-500/10 text-blue-400 border-blue-500/30',
      'Chemistry': 'bg-green-500/10 text-green-400 border-green-500/30',
      'Biology': 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
      'Computer Science': 'bg-purple-500/10 text-purple-400 border-purple-500/30',
      'Other': 'bg-gray-500/10 text-gray-400 border-gray-500/30'
    };
    return colors[topic] || colors['Other'];
  };

const generateFlashcards = (note) => {
  const cards = [];
  
  // Generate flashcards from concepts
  if (note.concepts && note.concepts.length > 0) {
    note.concepts.forEach(concept => {
      // Find a section that mentions this concept
      const relatedSection = note.sections.find(s =>
        s.desc.toLowerCase().includes(concept.toLowerCase())
      );

      if (relatedSection) {
        cards.push({
          id: `concept-${cards.length}`,
          front: `What is ${concept}?`,
          back: relatedSection.desc,
          type: 'concept',
          subject: note.subject
        });
      }
    });
  }

  // Generate flashcards from sections (using NEW structure)
  if (note.sections && note.sections.length > 0) {
    note.sections.forEach((section, idx) => {
      if (section.desc && section.desc.length > 50) {
        cards.push({
          id: `section-${idx}`,
          front: `Explain: ${section.title}`,
          back: section.desc,  // NEW: using 'desc' instead of 'content'
          type: 'section',
          subject: note.subject
        });
      }
    });
  }

  return cards;
};

const startFlashcards = (note) => {
  if (!note.flashcards || note.flashcards.length === 0) {
    alert("No flashcards available for this note yet.");
    return;
  }

  setActiveNoteForFlashcards(note);
  setCurrentFlashcardIndex(0);
  setIsFlipped(false);
  setShowFlashcardModal(true);
};


const openFlashcards = (cards) => {
  setFlashcards(cards);
  setCurrentFlashcardIndex(0);
  setIsFlashcardFlipped(false);
  setShowFlashcards(true);
};


  const nextFlashcard = () => {
    if (currentFlashcardIndex < flashcards.length - 1) {
      setCurrentFlashcardIndex(prev => prev + 1);
      setIsFlashcardFlipped(false);
    }
  };

  const previousFlashcard = () => {
    if (currentFlashcardIndex > 0) {
      setCurrentFlashcardIndex(prev => prev - 1);
      setIsFlashcardFlipped(false);
    }
  };

  const shuffleFlashcards = () => {
    const shuffled = [...flashcards].sort(() => Math.random() - 0.5);
    setFlashcards(shuffled);
    setCurrentFlashcardIndex(0);
    setIsFlashcardFlipped(false);
  };

const downloadAllNotes = async (format) => {
    const filteredNotes = getFilteredNotes();
    if (filteredNotes.length === 0) {
      alert('No notes to download');
      return;
    }

    setDownloadStatus(prev => ({ ...prev, 'all': 'Preparing combined download...' }));

    setTimeout(() => {
      let filename = `All_Notes_${selectedSubject.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}`;

      if (format === 'pdf') {
        try {
          const doc = new jsPDF();
          const pageWidth = doc.internal.pageSize.getWidth();
          const pageHeight = doc.internal.pageSize.getHeight();
          const margin = 20;
          const maxWidth = pageWidth - (margin * 2);
          let yPosition = 20;

          // Helper function to clean text (scoped within this function)
          const cleanText = (text) => {
            if (!text) return '';
            return text
              .replace(/[^\x00-\x7F]/g, '')
              .replace(/\s+/g, ' ')
              .trim();
          };

          // Helper function to add text
          const addText = (text, fontSize, isBold = false, color = [0, 0, 0]) => {
            doc.setFontSize(fontSize);
            doc.setFont(undefined, isBold ? 'bold' : 'normal');
            doc.setTextColor(...color);

            const cleanedText = cleanText(text);
            const lines = doc.splitTextToSize(cleanedText, maxWidth);
            
            lines.forEach(line => {
              if (yPosition > pageHeight - 30) {
                doc.addPage();
                yPosition = 20;
              }
              doc.text(line, margin, yPosition);
              yPosition += fontSize * 0.5;
            });
          };

          // --- LOOP THROUGH ALL NOTES ---
          filteredNotes.forEach((note, index) => {
            // If this isn't the first note, add a new page before starting
            if (index > 0) {
              doc.addPage();
              yPosition = 20;
            }

            // 1. Add Title
            const topicColor = getSubjectRGBColor(note.subject);
            addText(note.filename, 20, true, topicColor);
            yPosition += 5;

            // 2. Add Line
            doc.setDrawColor(...topicColor);
            doc.setLineWidth(1);
            doc.line(margin, yPosition, pageWidth - margin, yPosition);
            yPosition += 10;

            // 3. Add Metadata
            addText(`Subject: ${note.subject} | Date: ${note.date} | Sections: ${note.totalSections}`, 10, false, [100, 100, 100]);
            yPosition += 10;

            // 4. Add Content Header
            addText('Content', 16, true, topicColor);
            yPosition += 5;

            // 5. Add Sections
            note.sections.forEach((section, idx) => {
              yPosition += 5;
              addText(`${idx + 1}. ${section.title}`, 14, true, [55, 65, 81]);
              yPosition += 3;
              addText(section.desc, 11, false, [75, 85, 99]);
              yPosition += 8;
            });
          });

          // Save the single combined PDF
          doc.save(`${filename}.pdf`);
          setDownloadStatus(prev => ({ ...prev, 'all': `✓ Downloaded ${filteredNotes.length} notes as PDF` }));

        } catch (error) {
          console.error('PDF generation error:', error);
          alert('Failed to generate combined PDF.');
        }

      } else if (format === 'txt') {
        // Combine all notes into one text file
        let combinedContent = '';
        filteredNotes.forEach((note, index) => {
          if (index > 0) {
            combinedContent += '\n\n' + '='.repeat(100) + '\n\n';
          }
          combinedContent += generateTextContent(note);
        });

        const blob = new Blob([combinedContent], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${filename}.txt`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        setDownloadStatus(prev => ({ ...prev, 'all': `✓ Downloaded ${filteredNotes.length} notes as TXT` }));
      
      } else if (format === 'docx') {
         // Note: Combining DOCX client-side is very difficult without a library like docx.js.
         // This fallback downloads them individually to avoid breaking the app.
         alert("Batch downloading DOCX combines is not supported directly. Downloading files individually.");
         filteredNotes.forEach((note, i) => {
            setTimeout(() => downloadNote('docx', note), i * 500);
         });
      }

      setTimeout(() => {
        setDownloadStatus(prev => {
          const newStatus = { ...prev };
          delete newStatus['all'];
          return newStatus;
        });
      }, 3000);
    }, 500);
  };

  const removeFile = (fileId) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files);
    }
  };

  const getFilteredNotes = () => {
    let filtered = processedNotes;
    if (selectedSubject !== 'All Subjects') {
      filtered = filtered.filter(note => note.subject === selectedSubject);
    }

    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(note =>
        note.filename.toLowerCase().includes(query) ||
        note.subject.toLowerCase().includes(query) ||
        note.preview.toLowerCase().includes(query)
      );
    }

    return filtered;
  };

  const getAvailableSubjects = () => {
    const subjects = new Set(processedNotes.map(n => n.subject));
    return ['All Subjects', ...Array.from(subjects).sort()];
  };

  const toggleTOCSection = (section) => {
    setExpandedTOC(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

const renderHomePage = () => {
    // Feature data for the cards
    const features = [
      {
        icon: <Brain size={24} />,
        title: "AI Analysis",
        desc: "Instantly extract key concepts and summaries from your documents."
      },
      {
        icon: <Zap size={24} />,
        title: "Smart Flashcards",
        desc: "Auto-generate study cards to master your material faster."
      },
      {
        icon: <Search size={24} />,
        title: "Deep Search",
        desc: "Find specific topics across all your notes in milliseconds."
      },
      {
        icon: <Download size={24} />,
        title: "Multi-Format",
        desc: "Export your organized knowledge to PDF, DOCX, or Text."
      }
    ];

    return (
      <div className="min-h-[calc(100vh-80px)] relative flex flex-col items-center justify-center px-6 overflow-hidden">
        {/* Background Grid Pattern */}
        <div className={`absolute inset-0 z-0 opacity-20 ${
          isDark 
            ? 'bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]' 
            : 'bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]'
        }`}></div>

        {/* Ambient Glow Effects */}
        <div className={`absolute top-20 left-20 w-72 h-72 rounded-full blur-[100px] opacity-30 pointer-events-none ${
          isDark ? 'bg-purple-600' : 'bg-purple-300'
        }`}></div>
        <div className={`absolute bottom-20 right-20 w-96 h-96 rounded-full blur-[100px] opacity-30 pointer-events-none ${
          isDark ? 'bg-blue-600' : 'bg-blue-300'
        }`}></div>

        {/* Main Content */}
        <div className="relative z-10 max-w-5xl mx-auto text-center mt-10">
          
          {/* Badge */}
          <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full mb-8 border backdrop-blur-sm ${
            isDark 
              ? 'bg-purple-900/30 border-purple-500/30 text-purple-200' 
              : 'bg-white/60 border-purple-200 text-purple-700 shadow-sm'
          }`}>
            <Sparkles size={16} className="animate-pulse" />
            <span className="text-sm font-semibold tracking-wide uppercase">AI-Powered Learning Engine</span>
          </div>
          
          {/* Hero Title */}
          <h1 className="text-6xl md:text-7xl font-extrabold mb-6 tracking-tight">
            <span className={isDark ? 'text-white' : 'text-gray-900'}>Turn Chaos into </span>
            <span className="bg-gradient-to-r from-purple-500 via-blue-500 to-purple-500 bg-clip-text text-transparent bg-[length:200%_auto] animate-gradient">
              Knowledge
            </span>
          </h1>
          
          <p className={`text-xl md:text-2xl max-w-2xl mx-auto mb-10 leading-relaxed ${
            isDark ? 'text-gray-400' : 'text-gray-600'
          }`}>
            Upload your raw notes, PDFs, or images. Let our AI organize, summarize, and create study tools for you instantly.
          </p>
          
          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-20">
            <button
              onClick={() => setCurrentPage('upload')}
              className="group relative px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl font-bold text-lg shadow-lg shadow-purple-500/30 hover:shadow-purple-500/50 hover:scale-105 transition-all duration-300 overflow-hidden"
            >
              <div className="absolute inset-0 bg-white/20 group-hover:translate-x-full transition-transform duration-500 skew-x-12 -translate-x-full"></div>
              <div className="flex items-center gap-3">
                <Upload size={22} />
                Start Organizing
              </div>
            </button>
            
            <button
              onClick={() => setCurrentPage('dashboard')}
              className={`px-8 py-4 rounded-xl font-bold text-lg flex items-center gap-3 transition-all duration-300 hover:scale-105 ${
                isDark 
                  ? 'bg-gray-800 text-white border border-gray-700 hover:bg-gray-700' 
                  : 'bg-white text-gray-700 border border-gray-200 hover:bg-gray-50 shadow-md'
              }`}
            >
              <LayoutDashboard size={22} />
              Go to Dashboard
            </button>
          </div>

          {/* Feature Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 text-left">
            {features.map((feature, idx) => (
              <div key={idx} className={`p-6 rounded-2xl border backdrop-blur-md transition-all duration-300 hover:-translate-y-1 ${
                isDark 
                  ? 'bg-gray-800/40 border-gray-700 hover:bg-gray-800/60 hover:border-purple-500/50' 
                  : 'bg-white/60 border-gray-200 hover:bg-white/80 hover:border-purple-300 shadow-sm hover:shadow-md'
              }`}>
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 ${
                  isDark ? 'bg-purple-500/20 text-purple-300' : 'bg-purple-100 text-purple-600'
                }`}>
                  {feature.icon}
                </div>
                <h3 className={`text-lg font-bold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  {feature.title}
                </h3>
                <p className={`text-sm leading-relaxed ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                  {feature.desc}
                </p>
              </div>
            ))}
          </div>
          
        </div>
      </div>
    );
  };

  const renderUploadPage = () => (
    <div className="min-h-screen px-6 py-12 flex flex-col items-center justify-center">
      <div className="max-w-4xl w-full mx-auto">
        <h1 className="text-6xl font-bold text-center mb-4 bg-gradient-to-r from-purple-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
          Upload Your Notes
        </h1>
        <p className={`text-lg text-center mb-16 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          Drop your handwritten notes and let AI organize them instantly
        </p>

        <div
          className={`rounded-2xl p-10 text-center backdrop-blur-sm transition-all duration-300 ${isDark
              ? 'bg-gray-800/50 border-2 border-dashed hover:bg-gray-800/70'
              : 'bg-white/50 border-2 border-dashed hover:bg-white/70'
          } ${dragActive 
              ? 'border-purple-500 bg-purple-500/10' 
              : isDark 
                ? 'border-gray-600' 
                : 'border-gray-300'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <Upload size={64} className={`mx-auto mb-6 ${isDark ? 'text-purple-400' : 'text-purple-600'}`} />
          <h3 className={`text-2xl font-semibold mb-3 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Drop your files here
          </h3>
          <p className={`mb-6 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            or click to browse
          </p>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,.pdf"
            onChange={handleFileInputChange}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isProcessing}
            className="px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-blue-700 transition-all duration-300 disabled:opacity-50"
          >
            {isProcessing ? 'Processing...' : 'Select Files'}
          </button>
          <p className={`mt-4 text-sm ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>
            Supported: PNG, JPEG, PDF (max 10MB)
          </p>
        </div>

        {uploadedFiles.length > 0 && (
          <div className="mt-12">
            <div className="flex items-center justify-between mb-6">
              <h3 className={`text-2xl font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Processing Files ({uploadedFiles.length})
              </h3>
              {showViewDashboardButton && (
                <button
                  onClick={() => setCurrentPage('dashboard')}
                  className="px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg font-semibold hover:from-green-700 hover:to-emerald-700 transition-all duration-300 flex items-center gap-2"
                >
                  <LayoutDashboard size={20} />
                  View Dashboard
                </button>
              )}
            </div>
            <div className="space-y-4">
              {uploadedFiles.map(file => (
                <div
                  key={file.id}
                  className={`p-6 rounded-xl backdrop-blur-sm transition-all ${isDark ? 'bg-gray-800/50' : 'bg-white shadow-sm'}`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <FileText size={24} className={isDark ? 'text-purple-400' : 'text-purple-600'} />
                      <div>
                        <p className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                          {file.name}
                        </p>
                        <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                          {file.size}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      {file.status === 'processing' && (
                        <Clock size={20} className="text-blue-400 animate-spin" />
                      )}
                      {file.status === 'completed' && (
                        <CheckCircle size={20} className="text-green-400" />
                      )}
                      {file.status === 'error' && (
                        <AlertCircle size={20} className="text-red-400" />
                      )}
                      {file.status === 'completed' && file.type === 'application/pdf' && file.previewUrl && (
                        <button
                          onClick={() => {
                            setPDFPreviewUrl(file.previewUrl);
                            setShowPDFPreview(true);
                          }}
                          className={`p-1 rounded transition ${isDark 
                              ? 'text-purple-400 hover:bg-gray-700' 
                              : 'text-purple-600 hover:bg-gray-200'
                          }`}
                          title="Preview PDF"
                        >
                          <Eye size={20} />
                        </button>
                      )}
                      <button
                        onClick={() => removeFile(file.id)}
                        className={`p-1 rounded hover:bg-gray-700 transition ${isDark ? 'text-gray-400' : 'text-gray-600'}`}
                      >
                        <X size={18} />
                      </button>
                    </div>
                  </div>

                  {file.status === 'processing' && (
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${file.progress}%` }}
                      ></div>
                    </div>
                  )}

                  {file.status === 'error' && (
                    <p className="text-sm text-red-400 mt-2">
                      {file.errorMessage || 'Failed to process file'}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );

const generateCombinedMarkdown = (notes, subject) => {
  let content = [];
  content.push(`# ${subject} - Complete Notes Collection\n`);
  content.push(`**Generated:** ${new Date().toLocaleString()}\n`);
  content.push(`**Total Documents:** ${notes.length}\n`);
  content.push('---\n');

  notes.forEach((note, idx) => {
    content.push(`\n## Document ${idx + 1}: ${note.filename}\n`);
    content.push(`**Subject:** ${note.subject} | **Date:** ${note.date}\n`);
    
    if (note.tableOfContents) {
      content.push('\n### 📑 Table of Contents\n');
      note.tableOfContents.entries.forEach(entry => {
        const indent = '  '.repeat(entry.level - 1);
        content.push(`${indent}- **${entry.title}** (Page ${entry.page})`);
      });
      content.push('');
    }

    content.push('\n### 📝 Content\n');
    note.sections.forEach(section => {
      const heading = '#'.repeat((section.level || 1) + 3);
      content.push(`\n${heading} ${section.title}`);
      if (section.topic) {
        content.push(`\n> **Topic:** ${section.topic}`);
      }
      if (section.page) {
        content.push(`> **Page:** ${section.page}\n`);
      }
      content.push(`${section.desc}\n`);
    });

    if (note.concepts && note.concepts.length > 0) {
      content.push('\n### 🔑 Key Concepts\n');
      note.concepts.forEach(concept => {
        content.push(`- \`${concept}\``);
      });
    }

    content.push('\n---\n');
  });

  return content.join('\n');
};

const renderDashboard = () => {
  const filteredNotes = getFilteredNotes();
  const availableSubjects = getAvailableSubjects();

  return (
    <div className="min-h-screen px-6 py-8">
      <div className="max-w-7xl mx-auto">
        {/* Header Section */}
        <div className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <h1 className="text-4xl md:text-5xl font-extrabold mb-3 tracking-tight">
              <span className={isDark ? 'text-white' : 'text-slate-800'}>My Organized </span>
              <span className="bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                Notes
              </span>
            </h1>
            <p className={`text-lg ${isDark ? 'text-gray-400' : 'text-slate-500'}`}>
              You have <span className="font-semibold text-purple-500">{filteredNotes.length}</span> {filteredNotes.length === 1 ? 'note' : 'notes'}
              {selectedSubject !== 'All Subjects' && ` in ${selectedSubject}`}
              {searchQuery && ' matching your search'}.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            {processedNotes.length > 0 && (
              <button
                onClick={() => {
                  const notes = selectedSubject === 'All Subjects' 
                    ? processedNotes 
                    : processedNotes.filter(n => n.subject === selectedSubject);
                  
                  if (notes.length === 0) {
                    alert('No notes found for this subject');
                    return;
                  }

                  const content = generateCombinedMarkdown(notes, selectedSubject);
                  const blob = new Blob([content], { type: 'text/plain' });
                  const url = URL.createObjectURL(blob);
                  const link = document.createElement('a');
                  link.href = url;
                  link.download = `${selectedSubject.replace(/\s+/g, '_')}_All_Notes.txt`;
                  document.body.appendChild(link);
                  link.click();
                  document.body.removeChild(link);
                  URL.revokeObjectURL(url);
                  
                  alert(`Downloaded ${notes.length} notes from ${selectedSubject}!`);
                }}
                disabled={filteredNotes.length === 0}
                className={`px-5 py-2.5 rounded-xl font-medium border transition-all duration-300 flex items-center gap-2 ${
                  filteredNotes.length === 0
                    ? isDark ? 'bg-gray-700 text-gray-500 border-gray-600 cursor-not-allowed' : 'bg-gray-200 text-gray-400 border-gray-300 cursor-not-allowed'
                    : isDark 
                      ? 'border-gray-600 text-gray-300 hover:bg-gray-800 hover:border-gray-500' 
                      : 'bg-white border-gray-200 text-slate-600 hover:bg-gray-50 hover:text-slate-900 shadow-sm'
                }`}
              >
                <Download size={18} />
                Download {selectedSubject === 'All Subjects' ? 'All' : selectedSubject}
              </button>
            )}
            <button
              onClick={() => setCurrentPage('upload')}
              className="px-5 py-2.5 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl font-medium hover:shadow-lg hover:shadow-purple-500/25 hover:scale-105 transition-all duration-300 flex items-center gap-2"
            >
              <Upload size={18} />
              Upload New Note
            </button>
          </div>
        </div>

        {/* Search and Filter Bar */}
        <div className="flex flex-col md:flex-row gap-4 mb-8">
          <div className="flex-1 relative group">
            <Search size={20} className={`absolute left-4 top-1/2 transform -translate-y-1/2 transition-colors ${isDark ? 'text-gray-500 group-focus-within:text-purple-400' : 'text-slate-400 group-focus-within:text-purple-500'}`} />
            <input
              type="text"
              placeholder="Search by title, subject, or content..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={`w-full pl-12 pr-4 py-3.5 rounded-xl transition-all duration-300 outline-none border ${
                isDark
                  ? 'bg-gray-800/50 text-white border-gray-700 focus:border-purple-500 focus:bg-gray-800'
                  : 'bg-white text-slate-900 border-gray-200 shadow-sm focus:border-purple-500 focus:ring-2 focus:ring-purple-100'
              }`}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className={`absolute right-4 top-1/2 transform -translate-y-1/2 p-1 rounded-full transition-colors ${
                  isDark ? 'hover:bg-gray-700 text-gray-400 hover:text-white' : 'hover:bg-gray-200 text-gray-500 hover:text-gray-900'
                }`}
              >
                <X size={16} />
              </button>
            )}
          </div>
          <div className="relative">
            <select
              value={selectedSubject}
              onChange={(e) => setSelectedSubject(e.target.value)}
              className={`appearance-none w-full md:w-64 px-4 py-3.5 rounded-xl transition-all duration-300 outline-none cursor-pointer border ${
                isDark
                  ? 'bg-gray-800/50 text-white border-gray-700 hover:bg-gray-800'
                  : 'bg-white text-slate-900 border-gray-200 shadow-sm hover:border-purple-300'
              }`}
            >
              {availableSubjects.map(subject => (
                <option key={subject} value={subject}>{subject}</option>
              ))}
            </select>
            <ChevronDown size={16} className={`absolute right-4 top-1/2 transform -translate-y-1/2 pointer-events-none ${isDark ? 'text-gray-400' : 'text-slate-400'}`} />
          </div>
        </div>

        {/* Content Grid */}
        {filteredNotes.length === 0 ? (
          <div className={`flex flex-col items-center justify-center py-24 rounded-3xl border-2 border-dashed transition-all ${
            isDark 
              ? 'bg-gray-800/20 border-gray-700' 
              : 'bg-slate-50/50 border-slate-200'
          }`}>
            <div className={`p-6 rounded-full mb-6 ${isDark ? 'bg-gray-800' : 'bg-white shadow-sm'}`}>
              <FileText size={48} className={isDark ? 'text-gray-600' : 'text-slate-300'} />
            </div>
            <h3 className={`text-2xl font-bold mb-3 ${isDark ? 'text-white' : 'text-slate-800'}`}>
              No notes found
            </h3>
            <p className={`text-center max-w-md mb-8 ${isDark ? 'text-gray-400' : 'text-slate-500'}`}>
              {processedNotes.length === 0 
                ? 'Upload your first note to get started!'
                : searchQuery 
                  ? `No results for "${searchQuery}"`
                  : `No notes in ${selectedSubject}`
              }
            </p>
            <button
              onClick={() => setCurrentPage('upload')}
              className="px-6 py-3 bg-purple-600 text-white rounded-xl font-semibold hover:bg-purple-700 transition-colors shadow-lg shadow-purple-500/20"
            >
              Upload Note Now
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredNotes.map(note => (
              <div
                key={note.id}
                className={`group relative flex flex-col justify-between rounded-2xl p-6 transition-all duration-300 hover:-translate-y-2 border ${
                  isDark 
                    ? 'bg-gray-800 border-gray-700 shadow-lg hover:shadow-purple-500/10' 
                    : 'bg-white border-transparent shadow-[0_10px_40px_rgba(0,0,0,0.03)] hover:shadow-[0_20px_50px_rgba(99,102,241,0.1)]'
                }`}
              >
                {/* Card Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1 min-w-0 pr-4">
                    <h3 className={`text-xl font-bold mb-2 truncate ${isDark ? 'text-white' : 'text-slate-800'}`}>
                      {note.filename}
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {/* Display SUBJECT instead of filename */}
                      <span className={`px-2.5 py-0.5 rounded-md text-xs font-semibold border ${getTopicColor(note.subject || 'Other')}`}>
                        {note.subject || 'General'}
                      </span>
                      <span className={`px-2.5 py-0.5 rounded-md text-xs font-medium ${isDark ? 'bg-gray-700 text-gray-400' : 'bg-slate-100 text-slate-500'}`}>
                        {note.date}
                      </span>
                    </div>
                  </div>
                </div>

                {/* TOC Preview Section */}
                {note.tableOfContents && (
                  <div className={`mb-4 p-3 rounded-xl text-sm ${isDark ? 'bg-gray-900/50' : 'bg-slate-50'}`}>
                    <button
                      onClick={() => toggleTOCSection(note.id)}
                      className={`w-full flex items-center justify-between font-medium ${isDark ? 'text-blue-400' : 'text-blue-600'}`}
                    >
                      <span className="flex items-center gap-2">
                        <BookOpen size={16} />
                        Table of Contents ({note.tableOfContents.total_sections} sections)
                      </span>
                      {expandedTOC[note.id] ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                    </button>
                    
                    {expandedTOC[note.id] && (
                      <div className="mt-3 space-y-1.5 pl-1 max-h-32 overflow-y-auto scrollbar-thin">
                        {note.tableOfContents.entries.slice(0, 4).map((entry, idx) => (
                          <div key={idx} className="flex items-start gap-2">
                            <span className={`text-xs opacity-50 ${isDark ? 'text-gray-500' : 'text-slate-400'}`}>
                              {entry.id}.
                            </span>
                            <div className="flex-1 min-w-0">
                              <p className={`text-xs truncate ${isDark ? 'text-gray-300' : 'text-slate-700'}`}>
                                {entry.title}
                              </p>
                              <div className="flex items-center gap-2 mt-0.5">
                                <span className={`text-[10px] px-1.5 py-0.5 rounded border ${getTopicColor(entry.topic)}`}>
                                  {entry.topic}
                                </span>
                                <span className={`text-[10px] ${isDark ? 'text-gray-500' : 'text-slate-400'}`}>
                                  Page {entry.page}
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                        {note.tableOfContents.entries.length > 4 && (
                          <div className={`text-xs italic pt-1 ${isDark ? 'text-gray-500' : 'text-slate-400'}`}>
                            +{note.tableOfContents.entries.length - 4} more sections...
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Study Outline Section */}
                {note.sections && note.sections.length > 0 && (
                  <div className="mb-4">
                    <h4 className={`text-sm font-semibold mb-2 flex items-center gap-2 ${isDark ? 'text-white' : 'text-slate-800'}`}>
                      <LayoutDashboard size={14} className={isDark ? 'text-indigo-400' : 'text-indigo-600'} />
                      Study Outline
                    </h4>
                    <div className={`p-3 rounded-xl text-xs space-y-2 ${isDark ? 'bg-gray-900/30' : 'bg-slate-50'}`}>
                      {note.sections.slice(0, 2).map((section, idx) => (
                        <div key={idx} className="border-l-2 border-indigo-500 pl-2">
                          <p className={`font-medium mb-1 ${isDark ? 'text-gray-200' : 'text-slate-800'}`}>
                            {section.title}
                          </p>
                          <p className={`line-clamp-2 ${isDark ? 'text-gray-400' : 'text-slate-600'}`}>
                            {section.desc}
                          </p>
                        </div>
                      ))}
                      {note.sections.length > 2 && (
                        <p className={`text-[10px] italic ${isDark ? 'text-gray-500' : 'text-slate-400'}`}>
                          +{note.sections.length - 2} more sections
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {/* Key Concepts */}
                {note.concepts && note.concepts.length > 0 && (
                  <div className="mb-6 flex-1">
                    <h4 className={`text-sm font-semibold mb-2 flex items-center gap-2 ${isDark ? 'text-white' : 'text-slate-800'}`}>
                      <Tag size={14} className={isDark ? 'text-purple-400' : 'text-purple-600'} />
                      Key Concepts
                    </h4>
                    <div className="flex flex-wrap gap-1.5">
                      {note.concepts.slice(0, 4).map((concept, idx) => (
                        <span
                          key={idx}
                          className={`px-2 py-1 rounded text-[10px] uppercase tracking-wider font-medium ${
                            isDark 
                              ? 'bg-emerald-900/20 text-emerald-400 border border-emerald-900/50' 
                              : 'bg-emerald-50 text-emerald-700 border border-emerald-100'
                          }`}
                        >
                          {concept}
                        </span>
                      ))}
                      {note.concepts.length > 4 && (
                        <span className={`px-2 py-1 rounded text-[10px] font-medium ${
                          isDark ? 'text-purple-400' : 'text-purple-600'
                        }`}>
                          +{note.concepts.length - 4}
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {/* Card Actions Footer - WITH FLASHCARD BUTTON */}
                <div className="flex items-center gap-2 mt-auto pt-4 border-t border-dashed border-gray-200 dark:border-gray-700">
                  
                  {/* View Note Button */}
                  <button
                    onClick={() => handlePreview(note)}
                    className={`flex-1 py-2 rounded-lg font-medium text-sm transition-colors flex items-center justify-center gap-2 ${
                      isDark 
                        ? 'bg-purple-600/10 text-purple-400 hover:bg-purple-600 hover:text-white' 
                        : 'bg-purple-50 text-purple-700 hover:bg-purple-600 hover:text-white'
                    }`}
                  >
                    <Eye size={16} />
                    View Note
                  </button>

                  {/* 🧠 FLASHCARD BUTTON - THE MISSING BRAIN ICON */}
                  <button
                    onClick={() => startFlashcards(note)}
                    className={`p-2.5 rounded-lg transition-all duration-300 ${
                      isDark 
                        ? 'bg-blue-500/10 text-blue-400 hover:bg-blue-500 hover:text-white border border-blue-500/30' 
                        : 'bg-blue-50 text-blue-600 hover:bg-blue-500 hover:text-white border border-blue-200'
                    }`}
                    title="Study with Flashcards"
                  >
                    <Brain size={18} />
                  </button>

                  {/* Download Dropdown with PDF, DOCX, TXT */}
                  <div className="relative group/dropdown">
                    <button className={`p-2.5 rounded-lg transition-colors ${
                      isDark 
                        ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' 
                        : 'bg-gray-100 text-slate-600 hover:bg-gray-200'
                    }`}>
                      <Download size={18} />
                    </button>
                    
                    {/* Hover Dropdown Menu */}
                    <div className="absolute bottom-full right-0 mb-2 w-40 hidden group-hover/dropdown:block z-20">
                      <div className={`p-1 rounded-xl shadow-xl border overflow-hidden ${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            downloadNote('pdf', note);
                          }}
                          className={`w-full text-left px-3 py-2 text-sm rounded-lg font-medium flex items-center gap-2 ${
                            isDark 
                              ? 'text-gray-300 hover:bg-gray-700 hover:text-white' 
                              : 'text-slate-600 hover:bg-slate-50 hover:text-purple-600'
                          }`}
                        >
                          <FileText size={14} className="text-red-500" />
                          PDF
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            downloadNote('docx', note);
                          }}
                          className={`w-full text-left px-3 py-2 text-sm rounded-lg font-medium flex items-center gap-2 ${
                            isDark 
                              ? 'text-gray-300 hover:bg-gray-700 hover:text-white' 
                              : 'text-slate-600 hover:bg-slate-50 hover:text-purple-600'
                          }`}
                        >
                          <FileText size={14} className="text-blue-500" />
                          DOCX
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            downloadNote('txt', note);
                          }}
                          className={`w-full text-left px-3 py-2 text-sm rounded-lg font-medium flex items-center gap-2 ${
                            isDark 
                              ? 'text-gray-300 hover:bg-gray-700 hover:text-white' 
                              : 'text-slate-600 hover:bg-slate-50 hover:text-purple-600'
                          }`}
                        >
                          <FileText size={14} className="text-gray-500" />
                          TXT
                        </button>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Download Status Message */}
                {downloadStatus[note.id] && (
                  <div className="absolute inset-x-0 bottom-0 bg-emerald-500/90 text-white text-xs py-1 text-center rounded-b-2xl backdrop-blur-sm animate-fade-in">
                    {downloadStatus[note.id]}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
 const renderFlashcardModal = () => {
    if (!showFlashcardModal || !activeNoteForFlashcards) return null;

    const cards = activeNoteForFlashcards.flashcards || [];
    const currentCard = cards[currentFlashcardIndex];

    const nextCard = () => {
      setIsFlipped(false);
      setTimeout(() => {
        setCurrentFlashcardIndex((prev) => (prev + 1) % cards.length);
      }, 300); // Wait for flip back
    };

    const prevCard = () => {
      setIsFlipped(false);
      setTimeout(() => {
        setCurrentFlashcardIndex((prev) => (prev - 1 + cards.length) % cards.length);
      }, 300);
    };

    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm transition-all duration-300">
        <div className="relative w-full max-w-2xl">
          
          {/* Close Button */}
          <button
            onClick={() => setShowFlashcardModal(false)}
            className="absolute -top-12 right-0 p-2 text-white/70 hover:text-white hover:bg-white/10 rounded-full transition-all"
          >
            <X size={24} />
          </button>

          {/* Progress Header */}
          <div className="flex items-center justify-between text-white mb-4 px-1">
            <h3 className="font-bold text-lg flex items-center gap-2">
              <Brain className="text-purple-400" /> 
              {activeNoteForFlashcards.filename}
            </h3>
            <span className="text-sm font-mono bg-white/10 px-3 py-1 rounded-full border border-white/10">
              Card {currentFlashcardIndex + 1} / {cards.length}
            </span>
          </div>

          {/* Progress Bar */}
          <div className="w-full bg-gray-700 h-1.5 rounded-full mb-8 overflow-hidden">
            <div 
              className="bg-gradient-to-r from-purple-500 to-blue-500 h-full transition-all duration-500 ease-out"
              style={{ width: `${((currentFlashcardIndex + 1) / cards.length) * 100}%` }}
            />
          </div>

          {/* 3D Flip Card Container */}
          <div 
            className="group relative w-full h-96 cursor-pointer perspective-1000"
            onClick={() => setIsFlipped(!isFlipped)}
          >
            <div 
              className={`relative w-full h-full transition-all duration-700 transform-style-3d ${isFlipped ? 'rotate-y-180' : ''}`}
              style={{ transformStyle: 'preserve-3d', transform: isFlipped ? 'rotateY(180deg)' : 'rotateY(0deg)' }}
            >
              {/* FRONT (Question) */}
              <div 
                className={`absolute inset-0 flex flex-col items-center justify-center p-8 rounded-3xl shadow-2xl border backface-hidden ${
                  isDark 
                    ? 'bg-gradient-to-br from-gray-800 to-gray-900 border-gray-700' 
                    : 'bg-white border-white'
                }`}
                style={{ backfaceVisibility: 'hidden' }}
              >
                <span className="text-xs font-bold uppercase tracking-widest text-purple-500 mb-4">Question</span>
                <h4 className={`text-2xl md:text-3xl font-bold text-center leading-relaxed ${isDark ? 'text-white' : 'text-slate-800'}`}>
                  {currentCard?.front}
                </h4>
                <p className={`mt-8 text-sm animate-pulse ${isDark ? 'text-gray-500' : 'text-slate-400'}`}>
                  Click to reveal answer
                </p>
              </div>

              {/* BACK (Answer) */}
              <div 
                className={`absolute inset-0 flex flex-col items-center justify-center p-8 rounded-3xl shadow-2xl border backface-hidden rotate-y-180 ${
                  isDark 
                    ? 'bg-gradient-to-br from-indigo-900/90 to-purple-900/90 border-indigo-500/30' 
                    : 'bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-100'
                }`}
                style={{ 
                  backfaceVisibility: 'hidden', 
                  transform: 'rotateY(180deg)' 
                }}
              >
                <span className="text-xs font-bold uppercase tracking-widest text-indigo-300 mb-4">Answer</span>
                <p className={`text-xl md:text-2xl font-medium text-center leading-relaxed ${isDark ? 'text-white' : 'text-indigo-900'}`}>
                  {currentCard?.back}
                </p>
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="flex justify-between items-center mt-8 px-4">
            <button 
              onClick={(e) => { e.stopPropagation(); prevCard(); }}
              className={`p-4 rounded-full transition-all ${isDark ? 'bg-gray-800 hover:bg-gray-700 text-white' : 'bg-white hover:bg-gray-50 text-slate-800'} shadow-lg`}
            >
              <ChevronLeft size={24} />
            </button>
            
            <button 
              onClick={(e) => { e.stopPropagation(); setIsFlipped(!isFlipped); }}
              className="px-8 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl font-semibold shadow-lg shadow-purple-500/25 transition-all transform hover:scale-105"
            >
              {isFlipped ? 'Show Question' : 'Show Answer'}
            </button>

            <button 
              onClick={(e) => { e.stopPropagation(); nextCard(); }}
              className={`p-4 rounded-full transition-all ${isDark ? 'bg-gray-800 hover:bg-gray-700 text-white' : 'bg-white hover:bg-gray-50 text-slate-800'} shadow-lg`}
            >
              <ChevronRight size={24} />
            </button>
          </div>

        </div>
      </div>
    );
  };

const renderPreviewModal = () => {
  if (!showPreview || !previewFile) return null;
  
  const isNote = previewFile.sections !== undefined;
  const isImage = previewFile.type && previewFile.type.startsWith('image/');
  const isPDF = previewFile.type === 'application/pdf';

  return (
    <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-y-auto">
      <div className={`max-w-5xl w-full my-8 rounded-2xl ${isDark ? 'bg-gray-800' : 'bg-white'}`}>
        {/* Header */}
        <div className={`sticky top-0 flex items-center justify-between p-6 border-b z-10 ${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
          <div className="flex items-center gap-3">
            <FileText className={isDark ? 'text-purple-400' : 'text-purple-600'} size={24} />
            <div>
              <h2 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                {previewFile.filename || previewFile.name}
              </h2>
              {previewFile.size && (
                <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                  {previewFile.size}
                </p>
              )}
            </div>
          </div>
          <button
            onClick={closePreview}
            className={`p-2 rounded-lg transition-all hover:scale-110 ${isDark ? 'hover:bg-gray-700 text-gray-400 hover:text-white' : 'hover:bg-gray-100 text-gray-600 hover:text-gray-900'}`}
          >
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className={`p-6 overflow-auto max-h-[calc(90vh-180px)] ${isDark ? 'bg-gray-800' : 'bg-gray-50'}`}>
          {isNote ? (
            // PROCESSED NOTE PREVIEW
            <div>
              <div className="mb-8">
                <div className="flex flex-wrap gap-3 mb-4">
                  <span className={`px-4 py-2 rounded-full text-sm font-medium border ${getTopicColor(previewFile.subject)}`}>
                    {previewFile.subject}
                  </span>
                  <span className={`px-4 py-2 rounded-lg ${isDark ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700'}`}>
                    📅 {previewFile.date}
                  </span>
                  <span className={`px-4 py-2 rounded-lg ${isDark ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700'}`}>
                    📝 {previewFile.totalSections} sections
                  </span>
                </div>
              </div>

              {/* Table of Contents */}
              {previewFile.tableOfContents && (
                <div className="mb-8">
                  <h3 className={`text-xl font-bold mb-4 flex items-center gap-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    <BookOpen size={20} className={isDark ? 'text-blue-400' : 'text-blue-600'} />
                    Table of Contents ({previewFile.tableOfContents.total_sections} sections)
                  </h3>
                  <div className={`p-4 rounded-xl ${isDark ? 'bg-gray-900/50 border border-gray-700' : 'bg-white border border-gray-200'}`}>
                    {previewFile.tableOfContents.entries.slice(0, 10).map((entry, idx) => (
                      <div
                        key={idx}
                        className={`flex items-start gap-2 p-2 rounded transition-colors text-sm ${
                          isDark ? 'hover:bg-gray-800' : 'hover:bg-gray-100'
                        }`}
                        style={{ paddingLeft: `${(entry.level - 1) * 1}rem` }}
                      >
                        <span className={`font-mono min-w-[2rem] ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>
                          {entry.id}.
                        </span>
                        <div className="flex-1">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className={`font-medium ${isDark ? 'text-gray-200' : 'text-gray-900'}`}>
                              {entry.title}
                            </span>
                            <span className={`px-2 py-0.5 border rounded text-xs ${getTopicColor(entry.topic)}`}>
                              {entry.topic}
                            </span>
                          </div>
                          <div className={`text-xs mt-1 ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>
                            Page {entry.page} • Level {entry.level}
                          </div>
                        </div>
                      </div>
                    ))}
                    {previewFile.tableOfContents.entries.length > 10 && (
                      <div className={`text-xs text-center py-2 ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>
                        +{previewFile.tableOfContents.entries.length - 10} more sections
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Content Sections */}
              <div>
                <h3 className={`text-xl font-bold mb-4 flex items-center gap-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  <LayoutDashboard size={20} className={isDark ? 'text-indigo-400' : 'text-indigo-600'} />
                  Study Outline
                </h3>
                <div className="space-y-4">
                  {previewFile.sections.map((section, idx) => (
                    <div
                      key={idx}
                      className={`p-6 rounded-xl ${isDark ? 'bg-gray-900/50 border border-gray-700' : 'bg-white border border-gray-200'}`}
                    >
                      <div className="flex items-center gap-2 mb-3 flex-wrap">
                        <h4 className={`text-lg font-bold ${isDark ? 'text-purple-400' : 'text-purple-600'}`}>
                          {section.title}
                        </h4>
                        {section.topic && (
                          <span className={`px-2 py-1 border rounded text-xs ${getTopicColor(section.topic)}`}>
                            {section.topic}
                          </span>
                        )}
                      </div>
                      {(section.level || section.page) && (
                        <div className={`flex items-center gap-2 text-xs mb-3 ${isDark ? 'text-gray-500' : 'text-gray-600'}`}>
                          {section.level && <span>Level: H{section.level}</span>}
                          {section.page && <span>• Page: {section.page}</span>}
                        </div>
                      )}
                      <p className={`leading-relaxed ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                        {section.desc}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Key Concepts */}
              {previewFile.concepts && previewFile.concepts.length > 0 && (
                <div className="mt-8">
                  <h3 className={`text-xl font-bold mb-4 flex items-center gap-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    <Tag size={20} className={isDark ? 'text-purple-400' : 'text-purple-600'} />
                    Key Concepts
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {previewFile.concepts.map((concept, idx) => (
                      <span key={idx} className={`px-3 py-2 rounded text-sm font-medium ${
                        isDark ? 'bg-gray-700/80 text-gray-200' : 'bg-gray-200 text-gray-700'
                      }`}>
                        {concept}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            // RAW FILE PREVIEW (IMAGE OR PDF)
            <div>
              {/* IMAGE PREVIEW */}
              {isImage && previewFile.previewUrl && (
                <div className="flex items-center justify-center">
                  <img
                    src={previewFile.previewUrl}
                    alt={previewFile.name}
                    className="max-w-full max-h-[70vh] rounded-lg shadow-xl object-contain"
                  />
                </div>
              )}

              {/* PDF PREVIEW */}
              {isPDF && previewFile.previewUrl && (
                <div className="w-full h-[70vh]">
                  <iframe
                    src={previewFile.previewUrl}
                    className="w-full h-full rounded-lg"
                    title={previewFile.name}
                  />
                </div>
              )}

              {/* FALLBACK */}
              {!isImage && !isPDF && (
                <div className={`text-center py-12 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                  <FileText size={64} className="mx-auto mb-4 opacity-50" />
                  <p>Preview not available for this file type</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className={`flex items-center justify-between p-4 border-t ${isDark ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-gray-50'}`}>
          <div className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            {isNote ? 'Processed Note Preview' : isImage ? 'Image Preview' : isPDF ? 'PDF Preview' : 'File Preview'}
          </div>
          <button
            onClick={() => {
              if (previewFile.previewUrl) {
                const link = document.createElement('a');
                link.href = previewFile.previewUrl;
                link.download = previewFile.filename || previewFile.name;
                link.click();
              }
            }}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg font-semibold hover:bg-purple-700 transition-all flex items-center gap-2"
          >
            <Download size={16} />
            Download
          </button>
        </div>
      </div>
    </div>
  );
};

 return (
      <div className={`min-h-screen transition-colors duration-300 ${isDark 
  ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900' 
  : 'bg-[#f8fafc]' 
}`}>
      <nav className={`sticky top-0 z-40 backdrop-blur-md transition-all duration-300 ${
  isDark 
    ? 'bg-gray-900/80 border-b border-gray-800' 
    : 'bg-[#f1f5f9]/80 border-b border-slate-200' 
}`}>
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-8">
              <h1
                onClick={() => setCurrentPage('home')}
                className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent cursor-pointer hover:scale-105 transition"
              >
                NoteMap
              </h1>
              <div className="hidden md:flex gap-6">
                <button
                  onClick={() => setCurrentPage('home')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition ${currentPage === 'home'
                      ? isDark
                        ? 'bg-purple-600 text-white'
                        : 'bg-purple-600 text-white'
                      : isDark
                        ? 'text-gray-400 hover:text-white hover:bg-gray-800'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <Home size={18} />
                  Home
                </button>
                <button
                  onClick={() => setCurrentPage('upload')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition ${currentPage === 'upload'
                      ? isDark
                        ? 'bg-purple-600 text-white'
                        : 'bg-purple-600 text-white'
                      : isDark
                        ? 'text-gray-400 hover:text-white hover:bg-gray-800'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <Upload size={18} />
                  Upload
                </button>
                <button
                  onClick={() => setCurrentPage('dashboard')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition ${currentPage === 'dashboard'
                      ? isDark
                        ? 'bg-purple-600 text-white'
                        : 'bg-purple-600 text-white'
                      : isDark
                        ? 'text-gray-400 hover:text-white hover:bg-gray-800'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <LayoutDashboard size={18} />
                  Dashboard
                </button>
              </div>
            </div>

            <button
              onClick={() => setIsDark(!isDark)}
              className={`p-3 rounded-lg transition ${isDark ? 'bg-gray-800 text-yellow-400 hover:bg-gray-700' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
            >
              {isDark ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </div>
        </div>
      </nav>

      {currentPage === 'home' && renderHomePage()}
      {currentPage === 'upload' && renderUploadPage()}
      {currentPage === 'dashboard' && renderDashboard()}

      {renderPreviewModal()}
      {renderFlashcardModal()}

      {showPDFPreview && pdfPreviewUrl && (
        <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className={`max-w-4xl w-full h-[90vh] rounded-2xl overflow-hidden ${isDark ? 'bg-gray-800' : 'bg-white'}`}>
            <div className={`flex items-center justify-between p-4 border-b ${isDark ? 'border-gray-700' : 'border-gray-200'}`}>
              <h2 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                PDF Preview
              </h2>
              <button
                onClick={() => {
                  setShowPDFPreview(false);
                  setPDFPreviewUrl(null);
                }}
                className={`p-2 rounded-lg transition ${isDark ? 'hover:bg-gray-700 text-gray-400' : 'hover:bg-gray-100 text-gray-600'}`}
              >
                <X size={24} />
              </button>
            </div>
            <iframe
              src={pdfPreviewUrl}
              className="w-full h-[calc(100%-64px)]"
              title="PDF Preview"
            />
          </div>
        </div>
      )}
    </div>
  );
}