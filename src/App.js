import React, { useState, useRef, useEffect } from 'react';
import { Home, Upload, LayoutDashboard, Sun, Moon, FileText, Search, Download, ChevronDown, ChevronRight, X, CheckCircle, Clock, AlertCircle, Eye, BookOpen, Tag, MapPin, Sparkles, Brain, Zap, ChevronLeft, Shuffle, LogOut } from 'lucide-react';
import jsPDF from 'jspdf';
import * as pdfjsLib from "pdfjs-dist";
import { Amplify } from 'aws-amplify';
import { Authenticator } from '@aws-amplify/ui-react';
import { fetchAuthSession } from 'aws-amplify/auth';
import { post, get } from 'aws-amplify/api';
import '@aws-amplify/ui-react/styles.css';

pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.js';
Amplify.configure({
  Auth: {
    Cognito: {
      userPoolId: 'us-east-1_Ebc76UBJa',
      userPoolClientId: '2r704fnnfa991bs0fjoj6v4am',
      loginWith: {
        oauth: {
          domain: 'us-east-1ebc76ubja.auth.us-east-1.amazoncognito.com',
          scopes: ['email', 'openid', 'profile'],
          redirectSignIn: ['http://localhost:3000/'],
          redirectSignOut: ['http://localhost:3000/'],
          responseType: 'code'
        }
      }
    }
  },
  API: {
    REST: {
      noteApi: {
        endpoint: 'https://your-api-gateway-url.amazonaws.com/prod',
        region: 'us-east-1'
      }
    }
  }
});

// 🔥 AUTHENTICATION HOOK - MUST BE OUTSIDE THE COMPONENT
const useAuth = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const storedUser = localStorage.getItem('mockUser');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
    setLoading(false);
  }, []);

  const signIn = async (email) => {
    const mockUser = {
      userId: `user-${Date.now()}`,
      email: email,
      name: email.split('@')[0]
    };
    localStorage.setItem('mockUser', JSON.stringify(mockUser));
    setUser(mockUser);
  };

  const signOut = () => {
    localStorage.removeItem('mockUser');
    setUser(null);
  };

  return { user, loading, signIn, signOut };
};

// 🔥 LOGIN SCREEN COMPONENT - MUST BE OUTSIDE THE COMPONENT
const LoginScreen = ({ onSignIn, isDark }) => {
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleGoogleSignIn = async () => {
    setIsLoading(true);
    setTimeout(() => {
      onSignIn('demo.user@gmail.com');
      setIsLoading(false);
    }, 1500);
  };

  const handleEmailSignIn = () => {
    if (!email) return;
    setIsLoading(true);
    setTimeout(() => {
      onSignIn(email);
      setIsLoading(false);
    }, 1000);
  };

  return (
    <div className={`min-h-screen flex items-center justify-center p-6 ${
      isDark 
        ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900' 
        : 'bg-gradient-to-br from-purple-50 via-blue-50 to-purple-50'
    }`}>
      <div className={`absolute top-20 left-20 w-72 h-72 rounded-full blur-[120px] opacity-20 pointer-events-none ${
        isDark ? 'bg-purple-600' : 'bg-purple-400'
      }`}></div>
      <div className={`absolute bottom-20 right-20 w-96 h-96 rounded-full blur-[120px] opacity-20 pointer-events-none ${
        isDark ? 'bg-blue-600' : 'bg-blue-400'
      }`}></div>

      <div className={`relative max-w-md w-full p-8 rounded-3xl backdrop-blur-xl border shadow-2xl ${
        isDark 
          ? 'bg-gray-800/50 border-gray-700' 
          : 'bg-white/80 border-white/20'
      }`}>
        <div className="text-center mb-8">
          <div className={`w-16 h-16 mx-auto mb-4 rounded-2xl flex items-center justify-center ${
            isDark ? 'bg-purple-600' : 'bg-purple-500'
          }`}>
            <Brain size={32} className="text-white" />
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent mb-2">
            Welcome to NoteMap
          </h1>
          <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Sign in to organize your notes with AI
          </p>
        </div>

        <button
          onClick={handleGoogleSignIn}
          disabled={isLoading}
          className={`w-full py-3.5 px-4 rounded-xl font-semibold flex items-center justify-center gap-3 mb-4 transition-all duration-300 ${
            isDark 
              ? 'bg-white text-gray-900 hover:bg-gray-100' 
              : 'bg-white text-gray-900 hover:bg-gray-50 shadow-md hover:shadow-lg'
          } ${isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'}`}
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24">
            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
          </svg>
          {isLoading ? 'Signing in...' : 'Continue with Google'}
        </button>

        <div className="relative my-6">
          <div className={`absolute inset-0 flex items-center ${isDark ? 'text-gray-600' : 'text-gray-400'}`}>
            <div className="w-full border-t border-current"></div>
          </div>
          <div className="relative flex justify-center text-sm">
            <span className={`px-4 ${isDark ? 'bg-gray-800 text-gray-400' : 'bg-white text-gray-600'}`}>
              Or continue with email
            </span>
          </div>
        </div>

        <div>
          <input
            type="email"
            placeholder="your.email@gmail.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && email && !isLoading) {
                handleEmailSignIn();
              }
            }}
            className={`w-full px-4 py-3.5 rounded-xl mb-4 outline-none transition-all border ${
              isDark 
                ? 'bg-gray-900/50 text-white border-gray-700 focus:border-purple-500' 
                : 'bg-gray-50 text-gray-900 border-gray-200 focus:border-purple-500 focus:ring-2 focus:ring-purple-100'
            }`}
            disabled={isLoading}
          />
          <button
            onClick={handleEmailSignIn}
            disabled={isLoading || !email}
            className={`w-full py-3.5 rounded-xl font-semibold transition-all duration-300 ${
              isLoading || !email
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 hover:scale-105'
            } text-white shadow-lg shadow-purple-500/25`}
          >
            {isLoading ? 'Signing in...' : 'Sign In'}
          </button>
        </div>

        <p className={`text-xs text-center mt-6 ${isDark ? 'text-gray-500' : 'text-gray-500'}`}>
          By signing in, you agree to our Terms of Service and Privacy Policy
        </p>
      </div>
    </div>
  );
};

// 🔥 MAIN APP COMPONENT
function NoteMapApp() {
  const [currentPage, setCurrentPage] = useState('home');
  const [isDark, setIsDark] = useState(true);
  const [selectedSubject, setSelectedSubject] = useState('All Subjects');
  const [notes, setNotes] = useState([]);
  const [expandedSections, setExpandedSections] = useState({});
  const [searchQuery, setSearchQuery] = useState('');
  const [previewFile, setPreviewFile] = useState(null);
  const [flashcardFile, setFlashcardFile] = useState(null);
  const [currentFlashcardIndex, setCurrentFlashcardIndex] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);
  const [isShuffled, setIsShuffled] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showPDFPreview, setShowPDFPreview] = useState(false);
  const [pdfPreviewUrl, setPDFPreviewUrl] = useState(null);
  const fileInputRef = useRef(null);
  const { user, loading, signIn, signOut } = useAuth();

  // Show login screen if not authenticated
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  if (!user) {
    return <LoginScreen onSignIn={signIn} isDark={isDark} />;
  }

  const SUBJECTS = [
    'All Subjects',
    'Mathematics',
    'Physics',
    'Chemistry',
    'Biology',
    'Computer Science',
    'History',
    'Literature',
    'Psychology',
    'Economics',
    'Philosophy',
    'Engineering',
    'Medicine',
    'Law',
    'Art',
    'Music',
    'Language',
    'Business',
    'Geography',
    'Political Science',
    'Other'
  ];

  const cleanExtractedText = (rawText) => {
    return rawText
      .replace(/\s+/g, ' ')
      .replace(/([.!?])\s+/g, '$1\n\n')
      .replace(/•\s+/g, '\n• ')
      .trim();
  };

  const toggleSection = (noteId, index) => {
    const key = `${noteId}-${index}`;
    setExpandedSections(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const saveNoteToBackend = async (noteData) => {
    try {
      const session = await fetchAuthSession();
      const token = session.tokens?.idToken?.toString();

      const response = await post({
        apiName: 'noteApi',
        path: '/notes',
        options: {
          headers: {
            Authorization: `Bearer ${token}`
          },
          body: noteData
        }
      });

      console.log('Note saved successfully:', response);
      return response;
    } catch (error) {
      console.error('Error saving note:', error);
      return null;
    }
  };

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    for (const file of files) {
      setUploadProgress(0);
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 10;
        });
      }, 100);

      try {
        const result = await extractTextFromFile(file);
        clearInterval(progressInterval);
        setUploadProgress(100);

        if (!result.success) {
          console.error('Failed to process file:', file.name);
          setTimeout(() => setUploadProgress(0), 1000);
          continue;
        }

        const newNote = {
          id: Date.now() + Math.random(),
          name: file.name,
          filename: file.name,
          type: file.type,
          size: file.size,
          uploadDate: new Date().toISOString(),
          status: 'Processed',
          subject: result.subject || 'Other',
          text: result.text,
          sections: result.sections || [],
          concepts: result.concepts || [],
          wordCount: result.wordCount || 0,
          charCount: result.charCount || 0,
          date: result.date || new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          previewUrl: URL.createObjectURL(file),
          rawFile: file
        };

        setNotes(prev => [newNote, ...prev]);
        await saveNoteToBackend(newNote);

        setTimeout(() => setUploadProgress(0), 1500);
      } catch (error) {
        console.error('Error processing file:', error);
        clearInterval(progressInterval);
        setUploadProgress(0);
      }
    }

    event.target.value = '';
  };

  const getUniqueSubjects = () => {
    const subjects = notes.map(note => note.subject).filter(Boolean);
    return ['All Subjects', ...new Set(subjects)];
  };

  const filteredNotes = notes.filter(note => {
    const matchesSubject = selectedSubject === 'All Subjects' || note.subject === selectedSubject;
    const matchesSearch = note.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         note.text?.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         note.concepts?.some(c => c.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesSubject && matchesSearch;
  });

  const deleteNote = (noteId) => {
    setNotes(prev => prev.filter(note => note.id !== noteId));
  };

  const downloadNote = (note) => {
    const content = `
${note.name}
${'='.repeat(note.name.length)}

Subject: ${note.subject}
Date: ${note.date}
Word Count: ${note.wordCount}

${note.sections.map(section => `
${section.title}
${'-'.repeat(section.title.length)}
${section.desc}
`).join('\n')}

Key Concepts:
${note.concepts.map(c => `• ${c}`).join('\n')}
    `.trim();

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${note.name.replace(/\.[^/.]+$/, '')}_processed.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadFlashcards = (note) => {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 20;
    const cardWidth = (pageWidth - 3 * margin) / 2;
    const cardHeight = 60;

    doc.setFontSize(16);
    doc.text(note.name + ' - Flashcards', pageWidth / 2, margin, { align: 'center' });

    let yPos = margin + 15;
    let cardCount = 0;

    note.sections.forEach((section, index) => {
      if (yPos + cardHeight > pageHeight - margin) {
        doc.addPage();
        yPos = margin;
      }

      const xPos = (cardCount % 2 === 0) ? margin : pageWidth / 2 + margin / 2;

      doc.setDrawColor(139, 92, 246);
      doc.setLineWidth(0.5);
      doc.roundedRect(xPos, yPos, cardWidth, cardHeight, 3, 3);

      doc.setFontSize(12);
      doc.setFont(undefined, 'bold');
      doc.text('Q:', xPos + 5, yPos + 10);

      doc.setFont(undefined, 'normal');
      const questionLines = doc.splitTextToSize(section.title, cardWidth - 15);
      doc.text(questionLines, xPos + 5, yPos + 20);

      doc.setFont(undefined, 'bold');
      doc.text('A:', xPos + 5, yPos + 35);

      doc.setFont(undefined, 'normal');
      const answerLines = doc.splitTextToSize(section.desc.substring(0, 100) + '...', cardWidth - 15);
      doc.text(answerLines, xPos + 5, yPos + 45);

      cardCount++;
      if (cardCount % 2 === 0) {
        yPos += cardHeight + 10;
      }
    });

    doc.save(`${note.name}_flashcards.pdf`);
  };

  const openPreview = (note) => {
    setPreviewFile(note);
  };

  const openFlashcards = (note) => {
    setFlashcardFile(note);
    setCurrentFlashcardIndex(0);
    setShowAnswer(false);
    setIsShuffled(false);
  };

  const shuffleFlashcards = () => {
    if (!flashcardFile) return;
    const shuffled = [...flashcardFile.sections].sort(() => Math.random() - 0.5);
    setFlashcardFile({ ...flashcardFile, sections: shuffled });
    setIsShuffled(true);
    setCurrentFlashcardIndex(0);
    setShowAnswer(false);
  };

  const nextCard = () => {
    if (currentFlashcardIndex < flashcardFile.sections.length - 1) {
      setCurrentFlashcardIndex(prev => prev + 1);
      setShowAnswer(false);
    }
  };

  const prevCard = () => {
    if (currentFlashcardIndex > 0) {
      setCurrentFlashcardIndex(prev => prev - 1);
      setShowAnswer(false);
    }
  };

  const renderHomePage = () => (
    <div className="max-w-7xl mx-auto px-6 py-12">
      <div className="text-center mb-16 relative">
        <div className={`absolute inset-0 flex items-center justify-center opacity-5 pointer-events-none`}>
          <Brain size={400} className={isDark ? 'text-purple-500' : 'text-purple-300'} />
        </div>
        <h1 className={`text-6xl font-bold mb-4 bg-gradient-to-r ${
          isDark ? 'from-purple-400 via-pink-400 to-blue-400' : 'from-purple-600 via-pink-600 to-blue-600'
        } bg-clip-text text-transparent relative z-10`}>
          Welcome to NoteMap
        </h1>
        <p className={`text-xl mb-8 ${isDark ? 'text-gray-400' : 'text-gray-600'} relative z-10`}>
          Transform your notes into organized knowledge maps with AI-powered insights
        </p>
        <div className="flex gap-4 justify-center relative z-10">
          <button
            onClick={() => setCurrentPage('upload')}
            className="px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl font-semibold hover:shadow-2xl hover:shadow-purple-500/50 hover:scale-105 transition-all flex items-center gap-2"
          >
            <Upload size={20} />
            Upload Your First Note
          </button>
          <button
            onClick={() => setCurrentPage('dashboard')}
            className={`px-8 py-4 rounded-xl font-semibold transition-all flex items-center gap-2 ${
              isDark 
                ? 'bg-gray-800 text-white hover:bg-gray-700 border border-gray-700' 
                : 'bg-white text-gray-900 hover:bg-gray-50 border border-gray-200 shadow-md'
            }`}
          >
            <LayoutDashboard size={20} />
            View Dashboard
          </button>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-8 mb-16">
        {[
          {
            icon: <Sparkles size={32} className="text-purple-400" />,
            title: 'AI-Powered Analysis',
            desc: 'Automatically extract key concepts, sections, and insights from your notes'
          },
          {
            icon: <Brain size={32} className="text-blue-400" />,
            title: 'Smart Organization',
            desc: 'Organize notes by subject with intelligent categorization'
          },
          {
            icon: <Zap size={32} className="text-pink-400" />,
            title: 'Interactive Flashcards',
            desc: 'Generate flashcards automatically for efficient studying'
          }
        ].map((feature, index) => (
          <div
            key={index}
            className={`p-8 rounded-2xl transition-all hover:scale-105 cursor-pointer ${
              isDark 
                ? 'bg-gray-800/50 backdrop-blur-sm border border-gray-700 hover:border-purple-500/50' 
                : 'bg-white border border-gray-200 hover:border-purple-300 shadow-lg hover:shadow-xl'
            }`}
          >
            <div className={`w-16 h-16 rounded-xl flex items-center justify-center mb-4 ${
              isDark ? 'bg-gray-700/50' : 'bg-gray-50'
            }`}>
              {feature.icon}
            </div>
            <h3 className={`text-xl font-bold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
              {feature.title}
            </h3>
            <p className={isDark ? 'text-gray-400' : 'text-gray-600'}>
              {feature.desc}
            </p>
          </div>
        ))}
      </div>

      <div className={`p-8 rounded-2xl ${
        isDark 
          ? 'bg-gradient-to-br from-purple-900/20 to-blue-900/20 border border-purple-500/20' 
          : 'bg-gradient-to-br from-purple-50 to-blue-50 border border-purple-200'
      }`}>
        <h2 className={`text-3xl font-bold mb-6 ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Getting Started
        </h2>
        <div className="space-y-4">
          {[
            { step: '1', title: 'Upload Your Notes', desc: 'Support for PDF, images, and text files' },
            { step: '2', title: 'AI Processing', desc: 'Automatic extraction of key concepts and sections' },
            { step: '3', title: 'Study & Review', desc: 'Use flashcards, search, and organize your knowledge' }
          ].map((item) => (
            <div key={item.step} className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-full bg-gradient-to-r from-purple-600 to-blue-600 flex items-center justify-center text-white font-bold flex-shrink-0">
                {item.step}
              </div>
              <div>
                <h3 className={`font-bold mb-1 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  {item.title}
                </h3>
                <p className={isDark ? 'text-gray-400' : 'text-gray-600'}>
                  {item.desc}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderUploadPage = () => (
    <div className="max-w-4xl mx-auto px-6 py-12">
      <div className="text-center mb-8">
        <h1 className={`text-4xl font-bold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Upload Your Notes
        </h1>
        <p className={`text-lg ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          Upload PDF, images, or text files to get started
        </p>
      </div>

      <div
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all hover:scale-[1.02] ${
          isDark 
            ? 'border-gray-700 bg-gray-800/50 hover:border-purple-500 hover:bg-gray-800' 
            : 'border-gray-300 bg-gray-50 hover:border-purple-400 hover:bg-purple-50'
        }`}
      >
        <Upload size={64} className={`mx-auto mb-4 ${isDark ? 'text-gray-600' : 'text-gray-400'}`} />
        <h3 className={`text-xl font-bold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Click to upload or drag and drop
        </h3>
        <p className={`mb-4 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          PDF, PNG, JPG, or TXT files (Max 10MB each)
        </p>
        <div className="flex gap-2 justify-center flex-wrap">
          {['PDF', 'Image', 'Text'].map(type => (
            <span key={type} className={`px-3 py-1 rounded-full text-sm ${
              isDark ? 'bg-gray-700 text-gray-300' : 'bg-gray-200 text-gray-700'
            }`}>
              {type}
            </span>
          ))}
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.png,.jpg,.jpeg,.txt"
        onChange={handleFileUpload}
        className="hidden"
      />

      {uploadProgress > 0 && uploadProgress < 100 && (
        <div className="mt-6">
          <div className={`h-2 rounded-full overflow-hidden ${isDark ? 'bg-gray-700' : 'bg-gray-200'}`}>
            <div
              className="h-full bg-gradient-to-r from-purple-600 to-blue-600 transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          <p className={`text-sm text-center mt-2 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            Processing... {uploadProgress}%
          </p>
        </div>
      )}

      {notes.length > 0 && (
        <div className="mt-12">
          <h2 className={`text-2xl font-bold mb-6 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Recently Uploaded
          </h2>
          <div className="space-y-4">
            {notes.slice(0, 5).map(note => (
              <div
                key={note.id}
                className={`p-4 rounded-xl transition-all ${
                  isDark 
                    ? 'bg-gray-800/50 border border-gray-700 hover:border-purple-500/50' 
                    : 'bg-white border border-gray-200 hover:border-purple-300 shadow-sm'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3 flex-1">
                    <FileText className={isDark ? 'text-purple-400' : 'text-purple-600'} />
                    <div className="flex-1">
                      <h3 className={`font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                        {note.name}
                      </h3>
                      <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                        {formatFileSize(note.size)} · {note.subject}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="text-green-500" size={20} />
                    <span className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                      Processed
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const renderDashboard = () => (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className={`text-4xl font-bold mb-6 ${isDark ? 'text-white' : 'text-gray-900'}`}>
          Your Notes Dashboard
        </h1>

        <div className="flex flex-col md:flex-row gap-4 mb-6">
          <div className={`flex-1 relative ${isDark ? 'bg-gray-800/50' : 'bg-white'} rounded-xl border ${
            isDark ? 'border-gray-700' : 'border-gray-200'
          }`}>
            <Search className={`absolute left-4 top-1/2 -translate-y-1/2 ${
              isDark ? 'text-gray-500' : 'text-gray-400'
            }`} size={20} />
            <input
              type="text"
              placeholder="Search notes, concepts, or topics..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={`w-full pl-12 pr-4 py-3 rounded-xl outline-none ${
                isDark 
                  ? 'bg-transparent text-white placeholder-gray-500' 
                  : 'bg-transparent text-gray-900 placeholder-gray-400'
              }`}
            />
          </div>

          <select
            value={selectedSubject}
            onChange={(e) => setSelectedSubject(e.target.value)}
            className={`px-4 py-3 rounded-xl outline-none border ${
              isDark 
                ? 'bg-gray-800 text-white border-gray-700' 
                : 'bg-white text-gray-900 border-gray-200'
            }`}
          >
            {getUniqueSubjects().map(subject => (
              <option key={subject} value={subject}>{subject}</option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className={`p-6 rounded-xl ${
            isDark 
              ? 'bg-gradient-to-br from-purple-900/30 to-purple-800/30 border border-purple-500/30' 
              : 'bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200'
          }`}>
            <FileText className={isDark ? 'text-purple-400' : 'text-purple-600'} size={32} />
            <h3 className={`text-3xl font-bold mt-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
              {notes.length}
            </h3>
            <p className={isDark ? 'text-gray-400' : 'text-gray-600'}>
              Total Notes
            </p>
          </div>

          <div className={`p-6 rounded-xl ${
            isDark 
              ? 'bg-gradient-to-br from-blue-900/30 to-blue-800/30 border border-blue-500/30' 
              : 'bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200'
          }`}>
            <Tag className={isDark ? 'text-blue-400' : 'text-blue-600'} size={32} />
            <h3 className={`text-3xl font-bold mt-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
              {new Set(notes.flatMap(n => n.concepts || [])).size}
            </h3>
            <p className={isDark ? 'text-gray-400' : 'text-gray-600'}>
              Unique Concepts
            </p>
          </div>

          <div className={`p-6 rounded-xl ${
            isDark 
              ? 'bg-gradient-to-br from-pink-900/30 to-pink-800/30 border border-pink-500/30' 
              : 'bg-gradient-to-br from-pink-50 to-pink-100 border border-pink-200'
          }`}>
            <BookOpen className={isDark ? 'text-pink-400' : 'text-pink-600'} size={32} />
            <h3 className={`text-3xl font-bold mt-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
              {getUniqueSubjects().length - 1}
            </h3>
            <p className={isDark ? 'text-gray-400' : 'text-gray-600'}>
              Subjects
            </p>
          </div>
        </div>
      </div>

      {filteredNotes.length === 0 ? (
        <div className={`text-center py-16 ${
          isDark ? 'bg-gray-800/30' : 'bg-gray-50'
        } rounded-2xl`}>
          <FileText size={64} className={`mx-auto mb-4 ${
            isDark ? 'text-gray-600' : 'text-gray-400'
          }`} />
          <h3 className={`text-xl font-bold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
            No notes found
          </h3>
          <p className={`mb-6 ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            {searchQuery || selectedSubject !== 'All Subjects' 
              ? 'Try adjusting your filters' 
              : 'Upload your first note to get started'}
          </p>
          <button
            onClick={() => setCurrentPage('upload')}
            className="px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl font-semibold hover:shadow-xl transition-all"
          >
            Upload Notes
          </button>
        </div>
      ) : (
        <div className="grid gap-6">
          {filteredNotes.map(note => (
            <div
              key={note.id}
              className={`rounded-2xl overflow-hidden transition-all hover:scale-[1.01] ${
                isDark 
                  ? 'bg-gray-800/50 border border-gray-700' 
                  : 'bg-white border border-gray-200 shadow-sm'
              }`}
            >
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                        {note.name}
                      </h3>
                      <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        isDark 
                          ? 'bg-purple-900/30 text-purple-300 border border-purple-500/30' 
                          : 'bg-purple-100 text-purple-700 border border-purple-200'
                      }`}>
                        {note.subject}
                      </span>
                    </div>
                    <div className={`flex items-center gap-4 text-sm ${
                      isDark ? 'text-gray-400' : 'text-gray-600'
                    }`}>
                      <span className="flex items-center gap-1">
                        <Clock size={14} />
                        {note.date}
                      </span>
                      <span>{note.wordCount} words</span>
                      <span>{formatFileSize(note.size)}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => openPreview(note)}
                      className={`p-2 rounded-lg transition ${
                        isDark 
                          ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' 
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                      title="Preview"
                    >
                      <Eye size={18} />
                    </button>
                    <button
                      onClick={() => openFlashcards(note)}
                      className={`p-2 rounded-lg transition ${
                        isDark 
                          ? 'bg-blue-900/30 text-blue-400 hover:bg-blue-900/50' 
                          : 'bg-blue-50 text-blue-600 hover:bg-blue-100'
                      }`}
                      title="Study Flashcards"
                    >
                      <Brain size={18} />
                    </button>
                    <button
                      onClick={() => downloadNote(note)}
                      className={`p-2 rounded-lg transition ${
                        isDark 
                          ? 'bg-green-900/30 text-green-400 hover:bg-green-900/50' 
                          : 'bg-green-50 text-green-600 hover:bg-green-100'
                      }`}
                      title="Download"
                    >
                      <Download size={18} />
                    </button>
                    <button
                      onClick={() => deleteNote(note.id)}
                      className={`p-2 rounded-lg transition ${
                        isDark 
                          ? 'bg-red-900/30 text-red-400 hover:bg-red-900/50' 
                          : 'bg-red-50 text-red-600 hover:bg-red-100'
                      }`}
                      title="Delete"
                    >
                      <X size={18} />
                    </button>
                  </div>
                </div>

                {note.sections && note.sections.length > 0 && (
                  <div className="space-y-2">
                    {note.sections.map((section, idx) => (
                      <div
                        key={idx}
                        className={`rounded-lg overflow-hidden ${
                          isDark ? 'bg-gray-900/50' : 'bg-gray-50'
                        }`}
                      >
                        <button
                          onClick={() => toggleSection(note.id, idx)}
                          className={`w-full px-4 py-3 flex items-center justify-between hover:bg-opacity-80 transition ${
                            isDark ? 'text-white' : 'text-gray-900'
                          }`}
                        >
                          <span className="font-semibold">{section.title}</span>
                          {expandedSections[`${note.id}-${idx}`] ? (
                            <ChevronDown size={18} />
                          ) : (
                            <ChevronRight size={18} />
                          )}
                        </button>
                        {expandedSections[`${note.id}-${idx}`] && (
                          <div className={`px-4 pb-4 ${
                            isDark ? 'text-gray-400' : 'text-gray-600'
                          }`}>
                            {section.desc}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {note.concepts && note.concepts.length > 0 && (
                  <div className="mt-4">
                    <h4 className={`text-sm font-semibold mb-2 ${
                      isDark ? 'text-gray-400' : 'text-gray-600'
                    }`}>
                      Key Concepts:
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {note.concepts.map((concept, idx) => (
                        <span
                          key={idx}
                          className={`px-3 py-1 rounded-full text-sm ${
                            isDark 
                              ? 'bg-gray-700 text-gray-300' 
                              : 'bg-gray-200 text-gray-700'
                          }`}
                        >
                          {concept}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const renderFlashcardModal = () => {
    if (!flashcardFile) return null;

    const currentCard = flashcardFile.sections[currentFlashcardIndex];
    const progress = ((currentFlashcardIndex + 1) / flashcardFile.sections.length) * 100;

    return (
      <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className={`max-w-2xl w-full rounded-2xl overflow-hidden ${
          isDark ? 'bg-gray-800' : 'bg-white'
        }`}>
          <div className={`p-6 border-b ${isDark ? 'border-gray-700' : 'border-gray-200'}`}>
            <div className="flex items-center justify-between mb-4">
              <h2 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Flashcards: {flashcardFile.name}
              </h2>
              <button
                onClick={() => setFlashcardFile(null)}
                className={`p-2 rounded-lg transition ${
                  isDark ? 'hover:bg-gray-700 text-gray-400' : 'hover:bg-gray-100 text-gray-600'
                }`}
              >
                <X size={24} />
              </button>
            </div>

            <div className="flex items-center justify-between text-sm mb-2">
              <span className={isDark ? 'text-gray-400' : 'text-gray-600'}>
                Card {currentFlashcardIndex + 1} of {flashcardFile.sections.length}
              </span>
              <button
                onClick={shuffleFlashcards}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition ${
                  isDark 
                    ? 'bg-purple-900/30 text-purple-400 hover:bg-purple-900/50' 
                    : 'bg-purple-50 text-purple-600 hover:bg-purple-100'
                }`}
              >
                <Shuffle size={16} />
                {isShuffled ? 'Shuffled' : 'Shuffle'}
              </button>
            </div>

            <div className={`h-2 rounded-full overflow-hidden ${
              isDark ? 'bg-gray-700' : 'bg-gray-200'
            }`}>
              <div
                className="h-full bg-gradient-to-r from-purple-600 to-blue-600 transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          <div className="p-8">
            <div
              onClick={() => setShowAnswer(!showAnswer)}
              className={`min-h-[300px] rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all hover:scale-[1.02] ${
                isDark 
                  ? 'bg-gradient-to-br from-purple-900/30 to-blue-900/30 border-2 border-purple-500/30' 
                  : 'bg-gradient-to-br from-purple-50 to-blue-50 border-2 border-purple-200'
              }`}
            >
              {!showAnswer ? (
                <>
                  <span className={`text-sm font-semibold mb-4 ${
                    isDark ? 'text-purple-400' : 'text-purple-600'
                  }`}>
                    QUESTION
                  </span>
                  <h3 className={`text-2xl font-bold text-center ${
                    isDark ? 'text-white' : 'text-gray-900'
                  }`}>
                    {currentCard.title}
                  </h3>
                  <p className={`text-sm mt-4 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                    Click to reveal answer
                  </p>
                </>
              ) : (
                <>
                  <span className={`text-sm font-semibold mb-4 ${
                    isDark ? 'text-blue-400' : 'text-blue-600'
                  }`}>
                    ANSWER
                  </span>
                  <p className={`text-lg text-center leading-relaxed ${
                    isDark ? 'text-gray-300' : 'text-gray-700'
                  }`}>
                    {currentCard.desc}
                  </p>
                  <p className={`text-sm mt-4 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                    Click to hide answer
                  </p>
                </>
              )}
            </div>
          </div>

          <div className={`flex items-center justify-between p-6 border-t ${
            isDark ? 'border-gray-700' : 'border-gray-200'
          }`}>
            <button
              onClick={prevCard}
              disabled={currentFlashcardIndex === 0}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold transition ${
                currentFlashcardIndex === 0
                  ? 'opacity-50 cursor-not-allowed'
                  : isDark
                  ? 'bg-gray-700 text-white hover:bg-gray-600'
                  : 'bg-gray-200 text-gray-900 hover:bg-gray-300'
              }`}
            >
              <ChevronLeft size={18} />
              Previous
            </button>

            <button
              onClick={() => downloadFlashcards(flashcardFile)}
              className={`px-4 py-2 rounded-lg font-semibold transition flex items-center gap-2 ${
                isDark 
                  ? 'bg-purple-900/30 text-purple-400 hover:bg-purple-900/50' 
                  : 'bg-purple-50 text-purple-600 hover:bg-purple-100'
              }`}
            >
              <Download size={18} />
              Download PDF
            </button>

            <button
              onClick={nextCard}
              disabled={currentFlashcardIndex === flashcardFile.sections.length - 1}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold transition ${
                currentFlashcardIndex === flashcardFile.sections.length - 1
                  ? 'opacity-50 cursor-not-allowed'
                  : 'bg-gradient-to-r from-purple-600 to-blue-600 text-white hover:shadow-lg'
              }`}
            >
              Next
              <ChevronRight size={18} />
            </button>
          </div>
        </div>
      </div>
    );
  };

  const renderPreviewModal = () => {
    if (!previewFile) return null;

    const isNote = previewFile.sections && previewFile.sections.length > 0;
    const isImage = previewFile.type?.startsWith('image/');
    const isPDF = previewFile.type === 'application/pdf';

    return (
      <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className={`max-w-4xl w-full max-h-[90vh] rounded-2xl overflow-hidden flex flex-col ${
          isDark ? 'bg-gray-800' : 'bg-white'
        }`}>
          {/* Header */}
          <div className={`flex items-center justify-between p-4 border-b ${
            isDark ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-gray-50'
          }`}>
            <div>
              <h2 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                {previewFile.name}
              </h2>
              <p className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                {previewFile.subject} · {formatFileSize(previewFile.size)}
              </p>
            </div>
            <button
              onClick={() => setPreviewFile(null)}
              className={`p-2 rounded-lg transition ${
                isDark ? 'hover:bg-gray-700 text-gray-400' : 'hover:bg-gray-100 text-gray-600'
              }`}
            >
              <X size={24} />
            </button>
          </div>

          {/* Content */}
          <div className={`flex-1 overflow-y-auto p-6 ${isDark ? 'bg-gray-900' : 'bg-white'}`}>
            {isNote ? (
              // PROCESSED NOTE PREVIEW
              <div className="max-w-3xl mx-auto space-y-6">
                {previewFile.sections.map((section, idx) => (
                  <div
                    key={idx}
                    className={`p-6 rounded-xl ${
                      isDark 
                        ? 'bg-gray-800/50 border border-gray-700' 
                        : 'bg-gray-50 border border-gray-200'
                    }`}
                  >
                    <h3 className={`text-xl font-bold mb-3 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                      {section.title}
                    </h3>
                    <p className={`leading-relaxed ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
                      {section.desc}
                    </p>
                  </div>
                ))}

                {previewFile.concepts && previewFile.concepts.length > 0 && (
                  <div>
                    <h3 className={`text-lg font-bold mb-3 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                      Key Concepts
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {previewFile.concepts.map((concept, idx) => (
                        <span
                          key={idx}
                          className={`px-3 py-1.5 rounded-full text-sm font-medium ${
                            isDark ? 'bg-gray-700/80 text-gray-200' : 'bg-gray-200 text-gray-700'
                          }`}
                        >
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
          <div className={`flex items-center justify-between p-4 border-t ${
            isDark ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-gray-50'
          }`}>
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
    <div className={`min-h-screen transition-colors duration-300 ${
      isDark 
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
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition ${
                    currentPage === 'home'
                      ? 'bg-purple-600 text-white'
                      : isDark ? 'text-gray-400 hover:text-white hover:bg-gray-800' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <Home size={18} />
                  Home
                </button>
                <button
                  onClick={() => setCurrentPage('upload')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition ${
                    currentPage === 'upload'
                      ? 'bg-purple-600 text-white'
                      : isDark ? 'text-gray-400 hover:text-white hover:bg-gray-800' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <Upload size={18} />
                  Upload
                </button>
                <button
                  onClick={() => setCurrentPage('dashboard')}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition ${
                    currentPage === 'dashboard'
                      ? 'bg-purple-600 text-white'
                      : isDark ? 'text-gray-400 hover:text-white hover:bg-gray-800' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <LayoutDashboard size={18} />
                  Dashboard
                </button>
              </div>
            </div>

            {/* 🔥 USER INFO AND LOGOUT */}
            <div className="flex items-center gap-3">
              <div className={`hidden md:flex items-center gap-2 px-3 py-2 rounded-lg ${
                isDark ? 'bg-emerald-900/20 border border-emerald-500/30' : 'bg-emerald-50 border border-emerald-200'
              }`}>
                <CheckCircle size={16} className={isDark ? 'text-emerald-400' : 'text-emerald-600'} />
                <span className={`text-sm font-medium ${isDark ? 'text-emerald-300' : 'text-emerald-700'}`}>
                  {user.email}
                </span>
              </div>

              <button
                onClick={() => setIsDark(!isDark)}
                className={`p-3 rounded-lg transition ${
                  isDark ? 'bg-gray-800 text-yellow-400 hover:bg-gray-700' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {isDark ? <Sun size={20} /> : <Moon size={20} />}
              </button>

              <button
                onClick={signOut}
                className={`p-3 rounded-lg transition flex items-center gap-2 ${
                  isDark 
                    ? 'bg-red-900/20 text-red-400 hover:bg-red-900/40 border border-red-500/30' 
                    : 'bg-red-50 text-red-600 hover:bg-red-100 border border-red-200'
                }`}
                title="Sign Out"
              >
                <LogOut size={20} />
              </button>
            </div>
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
          <div className={`max-w-4xl w-full h-[90vh] rounded-2xl overflow-hidden ${
            isDark ? 'bg-gray-800' : 'bg-white'
          }`}>
            <div className={`flex items-center justify-between p-4 border-b ${
              isDark ? 'border-gray-700' : 'border-gray-200'
            }`}>
              <h2 className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                PDF Preview
              </h2>
              <button
                onClick={() => {
                  setShowPDFPreview(false);
                  setPDFPreviewUrl(null);
                }}
                className={`p-2 rounded-lg transition ${
                  isDark ? 'hover:bg-gray-700 text-gray-400' : 'hover:bg-gray-100 text-gray-600'
                }`}
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

// 🔥 HELPER FUNCTION - extractTextFromFile (MUST BE OUTSIDE COMPONENT)
const extractTextFromFile = async (file) => {
  try {
    let extractedText = '';
    
    // Handle different file types
    if (file.type === 'application/pdf') {
      const arrayBuffer = await file.arrayBuffer();
      const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
      
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const textContent = await page.getTextContent();
        const pageText = textContent.items.map(item => item.str).join(' ');
        extractedText += pageText + '\n';
      }
    } else if (file.type.startsWith('image/')) {
      // For images, we'll just return a placeholder
      // In a real app, you'd use OCR here
      extractedText = `[Image file: ${file.name}]\n\nThis is an image file. In a production app, OCR would be used to extract text from images.`;
    } else if (file.type === 'text/plain') {
      extractedText = await file.text();
    } else {
      return {
        success: false,
        error: 'Unsupported file type'
      };
    }

    // Clean the extracted text
    const cleanedText = extractedText
      .replace(/\s+/g, ' ')
      .replace(/([.!?])\s+/g, '$1\n\n')
      .trim();

    // Detect subject based on keywords
    const subjectKeywords = {
      'Mathematics': ['equation', 'theorem', 'integral', 'derivative', 'algebra', 'geometry', 'calculus'],
      'Physics': ['force', 'energy', 'momentum', 'velocity', 'quantum', 'relativity', 'mechanics'],
      'Chemistry': ['molecule', 'atom', 'reaction', 'compound', 'periodic', 'element', 'chemical'],
      'Biology': ['cell', 'organism', 'evolution', 'species', 'gene', 'protein', 'DNA'],
      'Computer Science': ['algorithm', 'programming', 'data structure', 'software', 'code', 'function'],
      'History': ['century', 'war', 'empire', 'revolution', 'ancient', 'medieval', 'civilization'],
      'Literature': ['novel', 'poem', 'author', 'character', 'narrative', 'metaphor', 'symbolism']
    };

    let detectedSubject = 'Other';
    let maxMatches = 0;

    for (const [subject, keywords] of Object.entries(subjectKeywords)) {
      const matches = keywords.filter(keyword => 
        cleanedText.toLowerCase().includes(keyword.toLowerCase())
      ).length;
      
      if (matches > maxMatches) {
        maxMatches = matches;
        detectedSubject = subject;
      }
    }

    // Extract sections (simple paragraph-based extraction)
    const paragraphs = cleanedText.split('\n\n').filter(p => p.trim().length > 50);
    const sections = paragraphs.slice(0, 5).map((para, idx) => ({
      title: `Section ${idx + 1}`,
      desc: para.substring(0, 200) + (para.length > 200 ? '...' : '')
    }));

    // Extract key concepts (simple word frequency analysis)
    const words = cleanedText.toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter(w => w.length > 5);
    
    const wordFreq = {};
    words.forEach(word => {
      wordFreq[word] = (wordFreq[word] || 0) + 1;
    });

    const concepts = Object.entries(wordFreq)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([word]) => word.charAt(0).toUpperCase() + word.slice(1));

    return {
      success: true,
      text: cleanedText,
      sections: sections.length > 0 ? sections : [{ title: 'Content', desc: cleanedText.substring(0, 500) }],
      concepts,
      subject: detectedSubject,
      wordCount: words.length,
      charCount: cleanedText.length,
      date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    };
  } catch (error) {
    console.error('Error extracting text from file:', error);
    return {
      success: false,
      error: error.message
    };
  }
};

export default NoteMapApp;