"""
NoteMap Backend - Enhanced OCR & NLP Processing (No OpenAI Required)

Install required packages:
pip install flask flask-cors pytesseract pillow PyPDF2 opencv-python numpy python-dotenv
pip install nltk spacy scikit-learn sentence-transformers keybert yake textstat

# Download spaCy model
python -m spacy download en_core_web_sm

For Tesseract (required):
Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
Linux: sudo apt-get install tesseract-ocr
Mac: brew install tesseract
"""
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Fix Windows encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import logging
import traceback
import json
import io
import re
from datetime import datetime
from collections import Counter, defaultdict

# Flask imports
from flask import Flask, request, jsonify
from flask_cors import CORS

# Image processing
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

# PDF processing
import PyPDF2

# OCR
import pytesseract

# NLP imports
import nltk
import spacy
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("⚠️ pdfplumber not installed - install with: pip install pdfplumber")

import PyPDF2
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("⚠️ pdf2image not installed - install with: pip install pdf2image")

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Advanced NLP
try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except:
    KEYBERT_AVAILABLE = False
    
try:
    import yake
    YAKE_AVAILABLE = True
except:
    YAKE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# IMPORTANT: Set your Tesseract path here (Windows only)
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('notemap_enhanced.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== INITIALIZATION ====================

# Download NLTK data
required_nltk = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
for package in required_nltk:
    try:
        nltk.download(package, quiet=True)
    except:
        logger.warning(f"Failed to download NLTK package: {package}")

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
try:
    STOP_WORDS = set(stopwords.words('english'))
except:
    STOP_WORDS = set(['the', 'is', 'at', 'which', 'on', 'and', 'or', 'this', 'that'])

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    logger.info("✓ spaCy loaded")
except:
    SPACY_AVAILABLE = False
    logger.warning("✗ spaCy not available - install with: python -m spacy download en_core_web_sm")

# Initialize KeyBERT
if KEYBERT_AVAILABLE:
    try:
        kw_model = KeyBERT()
        logger.info("✓ KeyBERT loaded")
    except:
        KEYBERT_AVAILABLE = False
        logger.warning("✗ KeyBERT initialization failed")

# Initialize Sentence Transformer
if SENTENCE_TRANSFORMERS_AVAILABLE:
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✓ Sentence-BERT loaded")
    except:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        logger.warning("✗ Sentence-BERT initialization failed")

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# ==================== TEXT CLEANING ====================

def clean_text(text):
    """Advanced text cleaning that PRESERVES structure and formatting"""
    
    # Detect if text contains code
    code_indicators = [
        r'public\s+class', r'#include', r'int\s+main', r'def\s+\w+\(',
        r'function\s+\w+', r'class\s+\w+:', r'import\s+\w+',
        r'var\s+\w+\s*=', r'const\s+\w+\s*=', r'let\s+\w+\s*=',
        r'for\s*\(', r'while\s*\(', r'if\s*\('
    ]
    
    is_code_heavy = any(re.search(pattern, text, re.IGNORECASE) for pattern in code_indicators)
    
    if is_code_heavy:
        logger.info("  Detected code content - preserving formatting")
        
        # Minimal cleaning for code
        text = re.sub(r'\r\n', '\n', text)  # Normalize line endings
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive blank lines
        text = re.sub(r'\s+\n', '\n', text)  # Clean trailing spaces but keep line breaks
        
        # Preserve spacing around operators and brackets
        text = re.sub(r'([{};()=<>+\-\*/])', r' \1 ', text)
        
        # Fix common OCR errors in code
        text = text.replace('|', 'l')  # Common in variable names
        text = text.replace('0O', '00')  # Zero vs O
        
        return text.strip()
    
    else:
        logger.info("  Detected regular text - applying smart cleaning")
        
        # IMPROVED: Preserve paragraph structure
        # Step 1: Normalize line endings
        text = re.sub(r'\r\n', '\n', text)
        
        # Step 2: Collapse spaces/tabs on same line (but keep newlines!)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Step 3: Remove trailing spaces before newlines
        text = re.sub(r' +\n', '\n', text)
        
        # Step 4: Limit to max 2 consecutive newlines (preserve paragraphs)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Step 5: Fix OCR bullet point merges
        text = re.sub(r'([^\n])•', r'\1\n•', text)  # Ensure • starts new line
        text = re.sub(r'([^\n])-\s+([A-Z])', r'\1\n- \2', text)  # Fix merged bullet lists
        
        # Step 6: Clean non-ASCII but preserve common symbols
        text = re.sub(r'[^\x00-\x7F\u00A0-\u024F\u2022\u2013\u2014]+', '', text)
        
        # Step 7: Fix hyphenation at line breaks
        text = text.replace('-\n', '')
        
        # Step 8: Fix punctuation spacing (but not around newlines)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = re.sub(r'\s+([.,!?;:])', r'\1', line)  # Remove space before punctuation
            line = re.sub(r'([.,!?;:])\s*([^\s\n])', r'\1 \2', line)  # Add space after
            line = re.sub(r'([.,!?;:])\1+', r'\1', line)  # Remove duplicate punctuation
            cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)
        
        # Step 9: Remove page numbers
        text = re.sub(r'\b[Pp]age\s+\d+\b', '', text)
        text = re.sub(r'\b\d+\s+of\s+\d+\b', '', text)
        
        # Step 10: Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
# ==================== IMAGE PREPROCESSING ====================

def detect_text_type(image_content):
    """Enhanced text type detection with stricter thresholds for handwriting"""
    try:
        image = Image.open(io.BytesIO(image_content))
        img_array = np.array(image.convert('L'))
        
        # Calculate multiple features
        edges = cv2.Canny(img_array, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        variance = np.var(img_array)
        
        # Calculate stroke width variation
        blur = cv2.GaussianBlur(img_array, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stroke_variation = 0
        if len(contours) > 10:
            areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
            if areas:
                stroke_variation = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
        
        # IMPROVED: Stricter decision logic favoring handwritten detection
        is_printed = (
            variance > 2500 and      # Increased from 2000
            edge_density < 0.12 and  # Decreased from 0.15
            stroke_variation < 1.5   # Decreased from 2.0
        )
        
        confidence = 0.85 if is_printed else 0.75  # Slightly higher confidence for handwritten
        text_type = "printed" if is_printed else "handwritten"
        
        logger.info(f"Text type: {text_type} (confidence: {confidence:.2f})")
        logger.info(f"  Variance: {variance:.0f}, Edges: {edge_density:.3f}, Stroke: {stroke_variation:.2f}")
        
        return is_printed, confidence
        
    except Exception as e:
        logger.warning(f"Detection failed: {str(e)}, assuming handwritten")
        return False, 0.5

def preprocess_image_for_ocr(image_content, is_printed=False):
    """
    ENHANCED: Advanced preprocessing with deskew + denoise + sharpen
    """
    try:
        logger.info("🔧 Advanced preprocessing started...")
        
        # Load image
        if isinstance(image_content, bytes):
            image = Image.open(io.BytesIO(image_content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_cv = image_content
        
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Denoise and remove shadows
        logger.info("  Denoising...")
        denoised = cv2.fastNlMeansDenoising(gray, h=15)
        
        # Step 2: Adaptive threshold (better than binary)
        logger.info("  Adaptive thresholding...")
        adaptive = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 35, 11
        )
        
        # Step 3: Deskew (rotate to horizontal alignment)
        logger.info("  Deskewing...")
        coords = np.column_stack(np.where(adaptive < 255))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            if abs(angle) > 0.5:  # Only deskew if needed
                (h, w) = adaptive.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                deskewed = cv2.warpAffine(adaptive, M, (w, h), 
                                          flags=cv2.INTER_CUBIC, 
                                          borderMode=cv2.BORDER_REPLICATE)
                logger.info(f"    Corrected angle: {angle:.2f}°")
            else:
                deskewed = adaptive
        else:
            deskewed = adaptive
        
        # Step 4: Sharpen text
        logger.info("  Sharpening...")
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        sharpened = cv2.filter2D(deskewed, -1, kernel)
        
        # Convert back to PIL
        preprocessed = Image.fromarray(sharpened)
        
        # Resize if too small
        width, height = preprocessed.size
        if width < 1000 or height < 1000:
            scale = max(1000 / width, 1000 / height)
            new_size = (int(width * scale), int(height * scale))
            preprocessed = preprocessed.resize(new_size, Image.LANCZOS)
            logger.info(f"  Upscaled to {new_size}")
        
        logger.info("✓ Preprocessing complete")
        return preprocessed
        
    except Exception as e:
        logger.warning(f"Preprocessing failed: {str(e)}")
        if isinstance(image_content, bytes):
            return Image.open(io.BytesIO(image_content))
        else:
            return Image.fromarray(image_content)
        
def normalize_extracted_text(text):
    """Cleans and standardizes extracted text while preserving structure"""
    
    # Fix common OCR duplications
    text = re.sub(r'(Amazon(\.|\s)+){2,}', 'Amazon.com ', text, flags=re.IGNORECASE)
    text = re.sub(r'(AWS\s+Shield\s+){2,}', 'AWS Shield ', text, flags=re.IGNORECASE)
    text = re.sub(r'(DDoS\s+){2,}', 'DDoS ', text, flags=re.IGNORECASE)
    
    # Collapse multiple spaces on same line (but keep newlines!)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Preserve paragraph breaks (max 2 newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Ensure bullet points start on new lines
    text = re.sub(r'([^\n])•', r'\1\n•', text)
    text = re.sub(r'([^\n])◦', r'\1\n◦', text)
    
    return text.strip()
# ==================== OCR ====================

def perform_tesseract_ocr(image_content, is_printed=False):
    """
    ENHANCED: Multi-PSM OCR with domain-specific normalization
    """
    try:
        logger.info("🔍 Multi-PSM OCR started...")
        
        preprocessed_image = preprocess_image_for_ocr(image_content, is_printed)
        
        # Multiple PSM configurations for best results
        configs = [
            r'--oem 3 --psm 6 -c preserve_interword_spaces=1',  # Uniform block
            r'--oem 3 --psm 4 -c preserve_interword_spaces=1',  # Single column
            r'--oem 3 --psm 3 -c preserve_interword_spaces=1',  # Fully automatic
        ]
        
        results = []
        for idx, cfg in enumerate(configs, 1):
            logger.info(f"  Trying PSM mode {idx}...")
            try:
                text = pytesseract.image_to_string(preprocessed_image, config=cfg, lang='eng')
                results.append(text)
                logger.info(f"    Extracted {len(text)} chars")
            except Exception as e:
                logger.warning(f"    PSM mode {idx} failed: {e}")
        
        # Choose the longest result (most complete)
        if not results:
            raise Exception("All OCR attempts failed")
        
        final_text = max(results, key=len)
        logger.info(f"  Best result: {len(final_text)} chars")
        
        # Domain-specific normalization (AWS/Tech terms)
        logger.info("  Normalizing technical terms...")
        replacements = {
            r'\bAws\b': 'AWS',
            r'\bDdos\b': 'DDoS',
            r'\bCom\b(?![a-z])': '.com',
            r'\bAmaz0n\b': 'Amazon',
            r'\bAmazon0\b': 'Amazon',
            r'—': '-',
            r'cid:': '',
            r'\bS3\b': 'S3',
            r'\bEc2\b': 'EC2',
            r'\bRds\b': 'RDS',
            r'\bVpc\b': 'VPC',
            r'\bIam\b': 'IAM',
        }
        
        for pattern, repl in replacements.items():
            final_text = re.sub(pattern, repl, final_text, flags=re.IGNORECASE)
        
        # Clean the text
        cleaned_text = clean_text(final_text)
        
        logger.info(f"✓ OCR complete: {len(cleaned_text)} chars")
        return cleaned_text
        
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        raise Exception(f"OCR failed: {str(e)}")
    
def extract_pdf_with_ocr_fallback(pdf_content):
    """
    ENHANCED: Hybrid PDF extraction with automatic OCR for scanned/image PDFs
    """
    try:
        logger.info("📄 Hybrid PDF extraction started...")
        
        # Step 1: Try pdfplumber first (best for text-based PDFs)
        if PDFPLUMBER_AVAILABLE:
            extracted_text = ""
            pages_with_text = 0
            pages_needing_ocr = []
            
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                num_pages = len(pdf.pages)
                logger.info(f"  PDF has {num_pages} pages")
                
                for i, page in enumerate(pdf.pages, 1):
                    try:
                        page_text = page.extract_text(
                            x_tolerance=1,
                            y_tolerance=3,
                            layout=True
                        )
                        
                        # Check if extraction was successful (more than 50 chars)
                        if page_text and len(page_text.strip()) > 50:
                            extracted_text += f"\n\n--- Page {i} ---\n\n{page_text.strip()}"
                            pages_with_text += 1
                            logger.info(f"  ✓ Page {i}: Direct extraction ({len(page_text)} chars)")
                        else:
                            pages_needing_ocr.append(i)
                            logger.info(f"  ⚠ Page {i}: Needs OCR (scanned/image)")
                    except Exception as e:
                        pages_needing_ocr.append(i)
                        logger.warning(f"  Page {i} extraction failed: {e}")
            
            # Step 2: If NO pages had extractable text, treat as fully scanned PDF
            if pages_with_text == 0:
                logger.info("📄 PDF appears to be fully scanned - using full OCR...")
                pages_needing_ocr = list(range(1, num_pages + 1))
        
        # Step 3: Apply OCR to pages that need it
        if pages_needing_ocr and PDF2IMAGE_AVAILABLE:
            logger.info(f"🔍 Applying OCR to {len(pages_needing_ocr)} pages...")
            
            try:
                # Convert PDF pages to images
                for page_num in pages_needing_ocr:
                    logger.info(f"  Processing page {page_num} with OCR...")
                    
                    # Convert single page to image
                    images = convert_from_bytes(
                        pdf_content, 
                        dpi=300,  # High quality
                        first_page=page_num, 
                        last_page=page_num
                    )
                    
                    for img in images:
                        # Convert PIL Image to bytes
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        buf.seek(0)
                        img_bytes = buf.read()
                        
                        # Detect if printed or handwritten
                        is_printed, confidence = detect_text_type(img_bytes)
                        logger.info(f"    Detected: {'printed' if is_printed else 'handwritten'} ({confidence:.2f})")
                        
                        # Perform OCR
                        ocr_text = perform_tesseract_ocr(img_bytes, is_printed=is_printed)
                        
                        if ocr_text.strip():
                            extracted_text += f"\n\n--- Page {page_num} (OCR) ---\n\n{ocr_text}"
                            logger.info(f"    ✓ Extracted {len(ocr_text)} chars")
                        else:
                            logger.warning(f"    ⚠ No text found on page {page_num}")
            
            except Exception as ocr_error:
                logger.error(f"OCR fallback failed: {ocr_error}")
                if not extracted_text.strip():
                    raise Exception(f"OCR processing failed: {ocr_error}")
        
        # Step 4: Check if we got any text at all
        if not extracted_text or len(extracted_text.strip()) < 50:
            raise Exception("No readable text extracted from PDF. May be corrupted or empty.")
        
        # Normalize the extracted text
        normalized_text = normalize_extracted_text(extracted_text)
        
        # Determine extraction method for reporting
        if pages_with_text > 0 and pages_needing_ocr:
            method = f"Hybrid PDF (Text: {pages_with_text}, OCR: {len(pages_needing_ocr)} pages)"
        elif pages_with_text > 0:
            method = "Direct PDF Text Extraction"
        else:
            method = "Full OCR Extraction (Scanned PDF)"
        
        logger.info(f"✓ {method}")
        logger.info(f"  Total extracted: {len(normalized_text)} characters")
        
        return normalized_text, method
        
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise Exception(f"Failed to extract PDF: {str(e)}")

# ==================== PDF EXTRACTION ====================

def extract_pdf_text(pdf_content):
    """Extract text from PDF preserving layout and code indentation"""
    try:
        logger.info("Processing PDF with layout-aware extraction (pdfplumber)...")
        
        text = ""
        pages_with_text = 0
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            num_pages = len(pdf.pages)
            
            if num_pages == 0:
                raise Exception("PDF has no pages")
            
            logger.info(f"PDF has {num_pages} pages")
            
            for i, page in enumerate(pdf.pages, 1):
                try:
                    # Extract with layout preservation for code formatting
                    layout_text = page.extract_text(
                        x_tolerance=1,      # Preserve horizontal spacing
                        y_tolerance=3,      # Preserve vertical spacing
                        layout=True         # Maintain layout structure
                    )
                    
                    if layout_text and len(layout_text.strip()) > 10:
                        text += f"\n\n--- Page {i} ---\n\n"
                        text += layout_text.strip() + "\n"
                        pages_with_text += 1
                        
                except Exception as e:
                    logger.warning(f"Page {i} extraction failed: {str(e)}")
                    continue
        
        if pages_with_text == 0:
            raise Exception("No text extracted - may be scanned/image-based PDF")
        
        # Apply smart cleaning (preserves code if detected)
        cleaned_text = clean_text(text)
        
        logger.info(f"✓ Extracted from {pages_with_text}/{num_pages} pages: {len(cleaned_text)} chars")
        logger.info(f"  Layout preservation: ENABLED")
        
        return cleaned_text
        
    except Exception as e:
        logger.error(f"pdfplumber extraction failed: {str(e)}")
        raise


# ==================== ADVANCED NLP ====================

def extract_keywords_advanced(text):
    """
    IMPROVED: Extract keywords using YAKE + KeyBERT + spaCy + TF-IDF
    Priority: YAKE (best) → KeyBERT → spaCy → TF-IDF
    """
    keywords = []
    
    # Method 1: YAKE (Unsupervised, no training required, works great!)
    if YAKE_AVAILABLE:
        try:
            kw_extractor = yake.KeywordExtractor(
                lan="en",           # Language
                n=2,                # Max ngram size (1-2 words)
                dedupLim=0.9,       # Deduplication threshold
                top=12,             # Number of keywords
                features=None
            )
            yake_keywords = kw_extractor.extract_keywords(text)
            # YAKE returns (keyword, score) - lower score is better
            keywords.extend([kw[0] for kw in yake_keywords])
            logger.info(f"YAKE extracted {len(yake_keywords)} keywords")
        except Exception as e:
            logger.warning(f"YAKE failed: {e}")
    
    # Method 2: KeyBERT (transformer-based, very good)
    if KEYBERT_AVAILABLE and len(keywords) < 8:
        try:
            kw_results = kw_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 2), 
                stop_words='english',
                top_n=12,
                use_mmr=True,      # Maximal Marginal Relevance for diversity
                diversity=0.7
            )
            keywords.extend([kw[0] for kw in kw_results])
            logger.info(f"KeyBERT extracted {len(kw_results)} keywords")
        except Exception as e:
            logger.warning(f"KeyBERT failed: {e}")
    
    # Method 3: spaCy NER + noun chunks (good for entities)
    if SPACY_AVAILABLE and len(keywords) < 8:
        try:
            doc = nlp(text[:10000])  # Process first 10k chars
            
            # Extract named entities
            entities = [ent.text for ent in doc.ents 
                       if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT', 'LAW']]
            keywords.extend(entities[:5])
            
            # Extract important noun chunks
            noun_chunks = [chunk.text for chunk in doc.noun_chunks 
                          if len(chunk.text.split()) <= 3 and len(chunk.text) > 3]
            keywords.extend(noun_chunks[:8])
            
            logger.info(f"spaCy extracted {len(entities)} entities and {len(noun_chunks)} noun chunks")
        except Exception as e:
            logger.warning(f"spaCy extraction failed: {e}")
    
    # Method 4: TF-IDF (classic, always works as fallback)
    if len(keywords) < 8:
        try:
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.85
            )
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores and sort by importance
            scores = tfidf_matrix.toarray()[0]
            word_scores = list(zip(feature_names, scores))
            word_scores.sort(key=lambda x: x[1], reverse=True)
            
            keywords.extend([word for word, score in word_scores[:12]])
            logger.info(f"TF-IDF extracted {len(feature_names)} keywords")
        except Exception as e:
            logger.warning(f"TF-IDF failed: {e}")
    
    # Clean and deduplicate
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen and len(kw) > 2:
            seen.add(kw_lower)
            unique_keywords.append(kw.title())  # Title case for consistency
    
    # Return top 10 most relevant
    final_keywords = unique_keywords[:10]
    logger.info(f"Final keywords: {final_keywords}")
    
    return final_keywords

def format_content_with_bullets(text, max_length=1000):
    """
    Format text content with bullet points for better readability
    Splits paragraphs into sentence-level bullets
    """
    if not text or len(text.strip()) < 20:
        return text
    
    # Split into sentences (preserving meaning)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Filter out empty or very short sentences
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not valid_sentences:
        return f"• {text[:max_length]}"
    
    # Format each sentence as a bullet point
    formatted_lines = []
    total_length = 0
    
    for sentence in valid_sentences:
        # Clean up the sentence
        sentence = sentence.strip()
        
        # Skip if it's just a heading or number
        if re.match(r'^[\d\.\-\)]+', sentence):
            continue
        
        # Add bullet point
        bullet_line = f"• {sentence}"
        
        # Check length limit
        if total_length + len(bullet_line) > max_length:
            formatted_lines.append("• [Content truncated...]")
            break
        
        formatted_lines.append(bullet_line)
        total_length += len(bullet_line)
    
    return "\n".join(formatted_lines) if formatted_lines else f"• {text[:max_length]}"


def smart_section_detection_enhanced_with_bullets(text):
    """
    ENHANCED: Multi-strategy section detection with bullet-formatted content
    This version formats all section descriptions with bullet points
    """
    
    # Strategy 1: Regex-based heading detection with IMPROVED patterns
    logger.info("Strategy 1: Improved regex-based heading detection...")
    sections = []
    
    lines = text.split('\n')
    current_section = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        is_heading = False
        heading_level = 1
        
        # ✅ IMPROVED: Stricter heading detection
        # Pattern 1: Explicit chapter/section markers
        if re.match(r'^(Chapter|Lecture|Section|Topic|Unit|Part)\s+\d+', line, re.IGNORECASE):
            is_heading = True
            heading_level = 1
        
        # Pattern 2: Numbered headings (1.1, 1.2.3, etc.)
        elif re.match(r'^\d+(\.\d+)*\s+[A-Z][A-Za-z\s]{2,}$', line):

            is_heading = True
            dots = line.split()[0].count('.')
            heading_level = min(dots + 1, 3)
        
        # Pattern 3: ALL CAPS short lines
        elif line.isupper() and 2 <= len(line.split()) <= 8:
            is_heading = True
            heading_level = 1
        
        # Pattern 4: Title Case followed by colon
        elif re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}:', line):

            is_heading = True
            heading_level = 1
        
        # Additional check: Standalone short line between blanks
        if not is_heading:
            prev_blank = (i == 0 or not lines[i-1].strip())
            next_blank = (i == len(lines)-1 or not lines[i+1].strip())
            if prev_blank and next_blank and len(line.split()) <= 8 and line[0].isupper():
                if not line[-1] in '.!?;,':
                    is_heading = True
                    heading_level = 1
        
        if is_heading:
            # Save previous section with bullet formatting
            if current_section and current_section['content'].strip():
                # Format content with bullet points
                formatted_content = format_content_with_bullets(current_section['content'])
                current_section['desc'] = formatted_content
                current_section['keywords'] = extract_keywords_advanced(current_section['content'])[:5]
                del current_section['content']
                sections.append(current_section)
            
            clean_title = re.sub(r'^[\d\.\)\]\s]+', '', line).strip()
            current_section = {
                'title': clean_title[:100],
                'level': heading_level,
                'content': '',
                'desc': '',
                'keywords': []
            }
        elif current_section is not None:
            current_section['content'] += ' ' + line
    
    # Save last section with bullet formatting
    if current_section and current_section['content'].strip():
        formatted_content = format_content_with_bullets(current_section['content'])
        current_section['desc'] = formatted_content
        current_section['keywords'] = extract_keywords_advanced(current_section['content'])[:5]
        del current_section['content']
        sections.append(current_section)
    
    if len(sections) >= 2:
        logger.info(f"✓ Strategy 1 succeeded: {len(sections)} sections with bullet formatting")
        return sections
    
    # Strategy 2: Semantic sectioning with bullet formatting
    logger.info("Strategy 2: Semantic sectioning with AI...")
    semantic_sections = semantic_sectioning(text)
    if semantic_sections and len(semantic_sections) >= 2:
        for section in semantic_sections:
            formatted_content = format_content_with_bullets(section['content'])
            section['desc'] = formatted_content
            section['keywords'] = extract_keywords_advanced(section['content'])[:5]
            del section['content']
        logger.info(f"✓ Strategy 2 succeeded: {len(semantic_sections)} sections")
        return semantic_sections
    
    # Strategy 3: Paragraph-based sectioning with bullet formatting
    logger.info("Strategy 3: Paragraph-based sectioning...")
    paragraph_sections = create_paragraph_based_sections(text)
    if paragraph_sections and len(paragraph_sections) >= 2:
        # Apply bullet formatting to existing descriptions
        for section in paragraph_sections:
            if 'desc' in section and section['desc']:
                section['desc'] = format_content_with_bullets(section['desc'])
        logger.info(f"✓ Strategy 3 succeeded: {len(paragraph_sections)} sections")
        return paragraph_sections
    
    # Strategy 4: Enhanced Main Content with bullet formatting
    logger.info("Strategy 4: Creating enhanced Main Content...")
    enhanced_main = create_summarized_main_content(text)
    # Apply bullet formatting
    for section in enhanced_main:
        if 'desc' in section and section['desc']:
            section['desc'] = format_content_with_bullets(section['desc'], max_length=1500)
    logger.info(f"✓ Strategy 4: Enhanced Main Content with bullets")
    return enhanced_main


def classify_subject_advanced(text):
    """Advanced subject classification using multiple signals"""
    
    # Subject keyword mapping
    SUBJECT_KEYWORDS = {
        'Computer Science': [
            'algorithm', 'data structure', 'programming', 'code', 'function', 
            'database', 'network', 'software', 'api', 'class', 'object', 'array',
            'python', 'java', 'javascript', 'sql', 'html', 'css', 'git', 'server'
        ],
        'Statistics': [
            'mean', 'median', 'variance', 'distribution', 'hypothesis', 'test',
            'probability', 'regression', 'correlation', 'sample', 'population',
            'confidence', 'interval', 'significance', 'chi-square', 'anova', 'p-value'
        ],
        'Mathematics': [
            'equation', 'theorem', 'proof', 'derivative', 'integral', 'matrix',
            'vector', 'algebra', 'calculus', 'geometry', 'trigonometry',
            'polynomial', 'logarithm', 'exponential', 'limit', 'function'
        ],
        'Management': [
            'organization', 'strategy', 'leadership', 'business', 'marketing',
            'finance', 'operations', 'management', 'planning', 'resources',
            'performance', 'decision', 'analysis', 'stakeholder', 'project'
        ],
        'Cybersecurity': [
            'security', 'encryption', 'firewall', 'malware', 'vulnerability',
            'attack', 'threat', 'defense', 'authentication', 'authorization',
            'cyber', 'hacker', 'breach', 'protection', 'risk', 'compliance'
        ]
    }
    
    text_lower = text.lower()
    scores = {}
    
    for subject, keywords in SUBJECT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[subject] = score
    
    if max(scores.values()) > 0:
        best_subject = max(scores, key=scores.get)
        confidence = min(95, 50 + (scores[best_subject] * 5))
        logger.info(f"Classified: {best_subject} (score: {scores[best_subject]}, confidence: {confidence}%)")
        return best_subject, confidence
    
    return "General", 50

def extract_sections(text):
    """
    ENHANCED: Extract structured sections with clean bullet point formatting
    
    Features:
    - Normalizes all unicode bullet symbols to standard '•'
    - Merges hyphenated words across line breaks
    - Joins broken sentences intelligently
    - Converts numbered/lettered lists to bullets
    - Conservative heading detection (reduces false positives)
    - Adds nested sub-bullets for better visual hierarchy
    
    Returns: list of sections with {'title', 'desc', 'level', 'page', 'topic'}
    """
    
    # Step 1: Normalize unicode bullet symbols to standard bullet (•)
    bullet_chars = r'[•●▪▫‣⁃⁌⁍\u2022\u2023\u25E6\u25AA\u25AB\u2024\u2043\u2219\uf0b7\uf0d8\uF0B7\u2027\u00B7\u25CF\u25CB\u25E6\u2043\u2219\u2024\u25AA\u25AB\u2027]'
    text = re.sub(r'[ \t]*' + bullet_chars, '•', text)
    
    # Step 2: Clean up whitespace and line breaks
    text = re.sub(r'\r', '\n', text)  # Normalize line endings
    text = re.sub(r'\t+', ' ', text)  # Replace tabs with spaces
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    
    lines = [ln.rstrip() for ln in text.split('\n')]
    
    # Step 3: Merge broken lines and hyphenated words
    merged_lines = []
    i = 0
    
    while i < len(lines):
        ln = lines[i].strip()
        
        if not ln:
            i += 1
            continue
        
        # Merge hyphenation at end of line (e.g., "multi-\nple" -> "multiple")
        if ln.endswith('-') and i + 1 < len(lines):
            next_ln = lines[i + 1].lstrip()
            if next_ln:  # Only merge if next line exists
                ln = ln[:-1] + next_ln  # Remove hyphen and join
                i += 1
                lines[i] = ln
                continue
        
        # Merge broken sentences: line ends without punctuation AND next line starts lowercase
        if i + 1 < len(lines):
            next_ln = lines[i + 1].lstrip()
            should_merge = (
                ln.endswith('...') or  # Ellipsis continuation
                (not re.search(r'[.!?]$', ln) and next_ln and next_ln[0].islower())  # Incomplete sentence
            )
            
            if should_merge:
                ln = ln + ' ' + next_ln
                i += 1
                lines[i] = ln
                continue
        
        merged_lines.append(ln)
        i += 1
    
    # Step 4: Extract sections with heading detection
    sections = []
    current = None
    
    def finish_current():
        """Finalize current section with bullet formatting"""
        if not current:
            return
        
        desc = current.get('desc', '').strip()
        if not desc:
            sections.append(current)
            return
        
        # Split into sentences
        try:
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(desc)
        except:
            # Fallback: naive sentence split
            sents = re.split(r'(?<=[.!?])\s+', desc)
        
        # Format each sentence as a bullet point
        normalized = []
        
        for s in sents:
            s = s.strip()
            if not s:
                continue
            
            # Convert numbered/lettered list markers to bullets
            # Matches: "1.", "a)", "[2]", "- item"
            if re.match(r'^[\(\[]?[a-zA-Z0-9]+[\.\)\]]\s*', s):
                s = re.sub(r'^[\(\[]?[a-zA-Z0-9]+[\.\)\]]\s*', '', s)
                s = '• ' + s.strip()
            elif re.match(r'^[\-\u2013\u2014]\s+', s):  # En-dash, em-dash
                s = re.sub(r'^[\-\u2013\u2014]\s+', '', s)
                s = '• ' + s.strip()
            elif s.startswith('•'):
                s = '• ' + s.lstrip('•').strip()
            else:
                # Regular sentence → also format as bullet
                s = '• ' + s
            
            normalized.append(s)
            
            # Add nested sub-bullet for visual hierarchy
            normalized.append('  •')  # Indented sub-bullet
        
        # Remove trailing empty sub-bullet
        if normalized and normalized[-1].strip() == '•':
            normalized.pop()
        
        # Join and truncate if too long
        formatted_desc = '\n'.join(normalized).strip()
        current['desc'] = formatted_desc[:2000]  # Safety limit
        sections.append(current)
    
    # Step 5: Process lines and detect headings
    for ln in merged_lines:
        line = ln.strip()
        if not line:
            continue
        
        # CONSERVATIVE heading detection (reduces false positives)
        is_heading = (
            re.match(r'^(Lecture|Chapter|Section|Topic|Unit|Part)\b', line, re.IGNORECASE) or
            re.match(r'^\d+(\.\d+)*\s+[A-Z][A-Za-z\s]{2,}$', line) or  # "1.2 Title Here"
            re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}:$', line) or  # "Introduction:"
            (line.isupper() and 2 <= len(line.split()) <= 10)  # ALL CAPS (2-10 words)
        )
        
        # Detect page markers: "--- Page N ---"
        page_match = re.match(r'---\s*Page\s*(\d+)\s*---', line, re.IGNORECASE)
        if page_match:
            if current and ('page' not in current or not current['page']):
                current['page'] = int(page_match.group(1))
            continue
        
        if is_heading:
            # Finalize previous section
            if current and current.get('desc'):
                finish_current()
            
            # Start new section
            current = {
                'title': line.rstrip(':')[:120],  # Remove trailing colon, limit length
                'desc': '',
                'level': 1,
                'page': None,
                'topic': None
            }
            continue
        
        # Not a heading → append to current section
        if current is None:
            # Create default section if no heading yet
            current = {
                'title': 'Main Content',
                'desc': line,
                'level': 1,
                'page': None,
                'topic': None
            }
        else:
            current['desc'] += ' ' + line
    
    # Finalize last section
    finish_current()
    
    # Step 6: Assign topics using classification
    for sec in sections:
        try:
            # Use your existing classify_subject_advanced function
            sec['topic'], _ = classify_subject_advanced(sec['desc'][:800])
        except:
            sec['topic'] = 'General'
        
        # Ensure page is at least 1
        if not sec.get('page'):
            sec['page'] = 1
    
    # Fallback: if no sections found, create one
    if not sections:
        try:
            topic, _ = classify_subject_advanced(text[:800])
        except:
            topic = 'General'
        
        sections = [{
            'title': 'Main Content',
            'desc': '• ' + text[:1000],
            'level': 1,
            'page': 1,
            'topic': topic
        }]
    
    logger.info(f"✓ Extracted {len(sections)} sections with clean bullet formatting")
    return sections


# ==============================================================================
# COPY THIS ENTIRE SECTION AND PASTE IT INTO YOUR app.py
# Place it AFTER the existing extract_keywords_advanced() function
# and BEFORE smart_section_detection()
# ==============================================================================

def detect_paragraph_boundaries(text):
    """Split text into meaningful paragraphs"""
    raw_paragraphs = text.split('\n\n')
    
    paragraphs = []
    for para in raw_paragraphs:
        para = para.strip()
        if len(para) > 50:
            paragraphs.append(para)
        elif len(para) > 500:
            sub_paras = [p.strip() for p in para.split('\n') if len(p.strip()) > 50]
            paragraphs.extend(sub_paras)
    
    return paragraphs

def detect_pseudo_headings(paragraph):
    """Detect if a paragraph is likely a heading"""
    heading_patterns = [
        r'^(chapter|section|unit|topic|lecture|part|introduction|conclusion|summary|overview|background)\s+\d+',
        r'^(chapter|section|unit|topic|lecture|part)\s+[ivxlcdm]+',
        r'^\d+\.\s*[A-Z]',
        r'^[A-Z][A-Z\s]{5,50}$',
        r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5}$',
    ]
    
    for pattern in heading_patterns:
        if re.match(pattern, paragraph, re.IGNORECASE):
            return True, paragraph[:80]
    
    words = paragraph.split()
    if len(words) <= 8 and len(paragraph) < 100:
        if not paragraph[-1] in '.!?,:;':
            if paragraph[0].isupper():
                return True, paragraph
    
    return False, None

def semantic_sectioning(text, min_section_length=300):
    """Use Sentence-BERT to detect topic boundaries"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        logger.info("Attempting semantic sectioning...")
        
        sentences = sent_tokenize(text)
        if len(sentences) < 5:
            return None
        
        sentences = sentences[:100]  # Performance limit
        embeddings = sentence_model.encode(sentences)
        
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
        
        threshold = np.percentile(similarities, 25)
        
        sections = []
        current_text = ""
        
        for i, sentence in enumerate(sentences):
            current_text += " " + sentence
            
            if i < len(similarities) and (similarities[i] < threshold or len(current_text) > 800):
                if len(current_text.strip()) > min_section_length:
                    title = generate_section_title(current_text)
                    sections.append({
                        'title': title,
                        'level': 1,
                        'content': current_text.strip()
                    })
                    current_text = ""
        
        if current_text.strip() and len(current_text.strip()) > min_section_length:
            title = generate_section_title(current_text)
            sections.append({
                'title': title,
                'level': 1,
                'content': current_text.strip()
            })
        
        if len(sections) >= 2:
            logger.info(f"Semantic sectioning found {len(sections)} sections")
            return sections
        
        return None
        
    except Exception as e:
        logger.warning(f"Semantic sectioning failed: {e}")
        return None

def generate_section_title(text, max_words=6):
    """Generate a descriptive title from content"""
    try:
        if KEYBERT_AVAILABLE:
            keywords = kw_model.extract_keywords(
                text[:500],
                keyphrase_ngram_range=(2, 3),
                stop_words='english',
                top_n=1
            )
            if keywords:
                title = keywords[0][0].title()
                if len(title.split()) <= max_words:
                    return title
        
        sentences = sent_tokenize(text)
        if sentences:
            first_sent = sentences[0]
            words = word_tokenize(first_sent)
            meaningful_words = [w for w in words if w.lower() not in STOP_WORDS and w.isalpha()]
            if meaningful_words:
                return ' '.join(meaningful_words[:max_words]).title()
        
        vectorizer = TfidfVectorizer(max_features=3, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([text[:500]])
        feature_names = vectorizer.get_feature_names_out()
        if len(feature_names) > 0:
            return ' '.join(feature_names).title()
        
        return "Content Section"
        
    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        return "Content Section"

def create_paragraph_based_sections(text):
    """Create sections from paragraphs"""
    logger.info("Creating paragraph-based sections...")
    
    paragraphs = detect_paragraph_boundaries(text)
    
    if len(paragraphs) < 2:
        return None
    
    sections = []
    current_section = None
    word_count = 0
    target_words = 350
    
    for para in paragraphs:
        is_heading, heading_text = detect_pseudo_headings(para)
        
        if is_heading:
            if current_section and current_section['content'].strip():
                current_section['desc'] = smart_summarize(current_section['content'], max_sentences=2)
                current_section['keywords'] = extract_keywords_advanced(current_section['content'])[:5]
                del current_section['content']
                sections.append(current_section)
            
            current_section = {
                'title': heading_text,
                'level': 1,
                'content': '',
                'keywords': []
            }
            word_count = 0
        else:
            para_words = len(para.split())
            
            if current_section is None:
                current_section = {
                    'title': f'Part {len(sections) + 1}',
                    'level': 1,
                    'content': para,
                    'keywords': []
                }
                word_count = para_words
            elif word_count + para_words > target_words and len(sections) < 8:
                current_section['desc'] = smart_summarize(current_section['content'], max_sentences=2)
                section_keywords = extract_keywords_advanced(current_section['content'])
                current_section['keywords'] = section_keywords[:5]
                
                if current_section['title'].startswith('Part '):
                    current_section['title'] = generate_section_title(current_section['content'])
                
                del current_section['content']
                sections.append(current_section)
                
                current_section = {
                    'title': f'Part {len(sections) + 1}',
                    'level': 1,
                    'content': para,
                    'keywords': []
                }
                word_count = para_words
            else:
                current_section['content'] += '\n\n' + para
                word_count += para_words
    
    if current_section and current_section['content'].strip():
        current_section['desc'] = smart_summarize(current_section['content'], max_sentences=2)
        section_keywords = extract_keywords_advanced(current_section['content'])
        current_section['keywords'] = section_keywords[:5]
        
        if current_section['title'].startswith('Part '):
            current_section['title'] = generate_section_title(current_section['content'])
        
        del current_section['content']
        sections.append(current_section)
    
    if len(sections) >= 2:
        logger.info(f"Created {len(sections)} paragraph-based sections")
        return sections
    
    return None

def create_summarized_main_content(text):
    """Enhanced Main Content with special handling for code-heavy documents"""
    logger.info("Creating enhanced Main Content section...")
    
    # Detect if document is code-heavy
    code_patterns = [
        r'public\s+class', r'#include', r'int\s+main', r'def\s+\w+\(',
        r'function\s+\w+', r'class\s+\w+:', r'import\s+\w+',
        r'for\s*\(', r'while\s*\(', r'if\s*\(', r'package\s+\w+;',
        r'<html>', r'<script>', r'SELECT\s+\*\s+FROM'
    ]
    
    code_matches = sum(1 for pattern in code_patterns if re.search(pattern, text, re.IGNORECASE))
    is_code_document = code_matches >= 3
    
    if is_code_document:
        logger.info("  Code-heavy document detected - using specialized extraction")
        
        # Extract code blocks and preserve formatting
        code_snippets = []
        
        # Try to find distinct code sections
        lines = text.split('\n')
        current_block = []
        in_code = False
        
        for line in lines:
            # Detect code lines (indented or containing code symbols)
            if line.strip() and (line.startswith('    ') or re.search(r'[{};()=]', line)):
                in_code = True
                current_block.append(line)
            elif in_code and not line.strip():
                # End of code block
                if len(current_block) > 2:
                    code_snippets.append('\n'.join(current_block))
                current_block = []
                in_code = False
        
        # Add last block if exists
        if current_block and len(current_block) > 2:
            code_snippets.append('\n'.join(current_block))
        
        # Extract programming language keywords for classification
        lang_keywords = {
            'Java': ['public class', 'private', 'void', 'import java'],
            'Python': ['def ', 'import ', 'class ', '__init__', 'self.'],
            'C++': ['#include', 'int main', 'std::', 'cout', 'namespace'],
            'JavaScript': ['function', 'const ', 'let ', 'var ', '=>'],
            'SQL': ['SELECT', 'FROM', 'WHERE', 'JOIN', 'INSERT']
        }
        
        detected_lang = "Code"
        for lang, keywords in lang_keywords.items():
            if sum(1 for kw in keywords if kw.lower() in text.lower()) >= 2:
                detected_lang = lang
                break
        
        # Build code-specific description
        preview_snippet = code_snippets[0][:500] if code_snippets else text[:500]
        
        enhanced_desc = f"💻 **{detected_lang} Code Document**\n\n"
        enhanced_desc += "This document contains programming code with preserved formatting.\n\n"
        enhanced_desc += f"**Preview:**\n```\n{preview_snippet}\n```\n\n"
        
        # Extract technical keywords
        keywords = extract_keywords_advanced(text)[:7]
        if keywords:
            enhanced_desc += f"📘 **Key Concepts:** {', '.join(keywords)}"
        
        return [{
            'title': f'{detected_lang} Code Notes',
            'level': 1,
            'desc': enhanced_desc,
            'keywords': [detected_lang, 'Programming', 'Code'] + keywords[:5],
            'entities': [],
            'code_blocks': len(code_snippets),
            'language': detected_lang
        }]
    
    else:
        # Original logic for regular text documents
        paragraphs = detect_paragraph_boundaries(text)
        
        summarized_paragraphs = []
        for para in paragraphs[:8]:
            if len(para) > 80:
                summary = smart_summarize(para, max_sentences=3, max_chars=200)
                summarized_paragraphs.append(summary)
        
        merged_summary = '\n\n'.join(summarized_paragraphs)
        
        keywords = extract_keywords_advanced(text)[:7]
        
        entities = []
        if SPACY_AVAILABLE:
            try:
                doc = nlp(text[:3000])
                entities = list(set([ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]))[:5]
            except:
                pass
        
        enhanced_desc = "📝 **Main Summary**\n\n" + merged_summary
        
        if keywords:
            enhanced_desc += f"\n\n📘 **Related Topics:** {', '.join(keywords)}"
        if entities:
            enhanced_desc += f"\n\n📌 **Key Mentions:** {', '.join(entities)}"
        
        return [{
            'title': 'Main Content',
            'level': 1,
            'desc': enhanced_desc,
            'keywords': keywords,
            'entities': entities
        }]


def smart_section_detection_enhanced(text):
    """ENHANCED: Multi-strategy section detection with stricter heading rules"""
    
    # Strategy 1: Regex-based heading detection with IMPROVED patterns
    logger.info("Strategy 1: Improved regex-based heading detection...")
    sections = []
    
    # CRITICAL FIX: Stronger heading patterns to avoid false positives
    lines = text.split('\n')
    current_section = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        is_heading = False
        heading_level = 1
        
        # ✅ IMPROVED: Stricter heading detection
        # Pattern 1: Explicit chapter/section markers
        if re.match(r'^(Chapter|Lecture|Section|Topic|Unit|Part)\s+\d+', line, re.IGNORECASE):
            is_heading = True
            heading_level = 1
        
        # Pattern 2: Numbered headings (1.1, 1.2.3, etc.)
        elif re.match(r'^\d+(\.\d+)*\s+[A-Z][A-Za-z\s]{2,}$', line):
            is_heading = True
            dots = line.split()[0].count('.')
            heading_level = min(dots + 1, 3)  # Max level 3
        
        # Pattern 3: ALL CAPS short lines (but not random sentences)
        elif line.isupper() and 2 <= len(line.split()) <= 8:
            is_heading = True
            heading_level = 1
        
        # Pattern 4: Title Case followed by colon (e.g., "Introduction:")
        elif re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}:\s*$', line):
            is_heading = True
            heading_level = 1
        
        # Additional check: Standalone short line between blanks (but not ending with punctuation)
        if not is_heading:
            prev_blank = (i == 0 or not lines[i-1].strip())
            next_blank = (i == len(lines)-1 or not lines[i+1].strip())
            if prev_blank and next_blank and len(line.split()) <= 8 and line[0].isupper():
                # NOT a heading if it ends with sentence punctuation
                if not line[-1] in '.!?;,':
                    is_heading = True
                    heading_level = 1
        
        if is_heading:
            if current_section and current_section['content'].strip():
                current_section['desc'] = smart_summarize(current_section['content'])
                current_section['keywords'] = extract_keywords_advanced(current_section['content'])[:5]
                del current_section['content']
                sections.append(current_section)
            
            clean_title = re.sub(r'^[\d\.\)\]\s]+', '', line).strip()
            current_section = {
                'title': clean_title[:100],
                'level': heading_level,
                'content': '',
                'desc': '',
                'keywords': []
            }
        elif current_section is not None:
            current_section['content'] += ' ' + line
    
    if current_section and current_section['content'].strip():
        current_section['desc'] = smart_summarize(current_section['content'])
        current_section['keywords'] = extract_keywords_advanced(current_section['content'])[:5]
        del current_section['content']
        sections.append(current_section)
    
    if len(sections) >= 2:
        logger.info(f"✓ Strategy 1 succeeded: {len(sections)} sections found")
        return sections
    
    # Strategy 2: Semantic sectioning with Sentence-BERT
    logger.info("Strategy 2: Semantic sectioning with AI...")
    semantic_sections = semantic_sectioning(text)
    if semantic_sections and len(semantic_sections) >= 2:
        for section in semantic_sections:
            section['desc'] = smart_summarize(section['content'])
            section['keywords'] = extract_keywords_advanced(section['content'])[:5]
            del section['content']
        logger.info(f"✓ Strategy 2 succeeded: {len(semantic_sections)} sections found")
        return semantic_sections
    
    # Strategy 3: Paragraph-based sectioning
    logger.info("Strategy 3: Paragraph-based sectioning...")
    paragraph_sections = create_paragraph_based_sections(text)
    if paragraph_sections and len(paragraph_sections) >= 2:
        logger.info(f"✓ Strategy 3 succeeded: {len(paragraph_sections)} sections found")
        return paragraph_sections
    
    # Strategy 4: Enhanced Main Content (fallback)
    logger.info("Strategy 4: Creating enhanced Main Content...")
    enhanced_main = create_summarized_main_content(text)
    logger.info(f"✓ Strategy 4: Enhanced Main Content with keywords and summaries")
    return enhanced_main


smart_section_detection = smart_section_detection_enhanced

def smart_summarize(text, max_sentences=3, max_chars=250):
    """
    Intelligent multi-strategy summarizer.
    ✅ Uses Sentence-BERT if available
    ✅ Falls back to TF-IDF or simple lead summary
    ✅ Keeps summaries short and coherent
    """
    try:
        # Quick exit for short text
        if not text or len(text.strip()) < max_chars:
            return text.strip()

        # Tokenize into sentences
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return ' '.join(sentences)

        summary = None

        # ===============================
        # 🧠 Strategy 1: Sentence-BERT (if available)
        # ===============================
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE and len(sentences) > 5:
                embeddings = sentence_model.encode(sentences)
                centroid = np.mean(embeddings, axis=0)
                similarities = cosine_similarity([centroid], embeddings)[0]

                # Pick top N sentences by similarity
                top_indices = np.argsort(similarities)[-max_sentences:]
                top_indices = sorted(top_indices)  # keep original order
                summary_sentences = [sentences[i] for i in top_indices]
                summary = ' '.join(summary_sentences)

                if len(summary) <= max_chars:
                    return summary
        except Exception as e:
            logger.warning(f"Sentence-BERT summarization failed: {e}")

        # ===============================
        # ✳️ Strategy 2: TF-IDF Weighted Selection
        # ===============================
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

            top_indices = np.argsort(scores)[-max_sentences:]
            top_indices = sorted(top_indices)

            summary_sentences = [sentences[i] for i in top_indices]
            summary = ' '.join(summary_sentences)

            if len(summary) <= max_chars:
                return summary
        except Exception as e:
            logger.warning(f"TF-IDF summarization failed: {e}")

        # ===============================
        # 🪄 Strategy 3: Simple fallback (first N sentences)
        # ===============================
        summary = ' '.join(sentences[:max_sentences])
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(' ', 1)[0] + '...'

        return summary

    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        return text[:max_chars].rsplit(' ', 1)[0] + '...' if len(text) > max_chars else text
    """Intelligent text summarization with structure preservation"""
    
    # If text contains bullet points or lists, preserve them
    has_bullets = bool(re.search(r'^\s*[•\-◦]\s+', text, re.MULTILINE))
    
    if has_bullets and len(text) < max_chars * 2:
        # For short bulleted lists, return as-is
        return text
    
    if len(text) < max_chars:
        return text
    
    # ... rest of existing code ...

def extract_concepts(text, max_keywords=8):
    """
    ENHANCED: Multi-strategy keyword extraction with noise filtering
    
    Priority:
    1. YAKE (unsupervised, no training needed)
    2. TF-IDF with smart filtering
    3. Fallback to frequency analysis
    
    Features:
    - Filters out institution names (VIT, university, etc.)
    - Removes common noise words
    - Returns capitalized, clean keywords
    
    Returns: list of up to max_keywords concepts
    """
    
    # Limit text size for performance
    text_short = text[:5000]
    
    # Define noise tokens to filter out
    noise_tokens = {
        'vit', 'university', 'vellore', 'dr', 'prof', 'professor',
        'department', 'lecture', 'slide', 'slides', 'page', 'ppt',
        'chapter', 'section', 'notes', 'study', 'exam', 'test',
        'document', 'content', 'main', 'introduction', 'conclusion'
    }
    
    # Strategy 1: YAKE (best for academic content)
    if YAKE_AVAILABLE:
        try:
            import yake
            kw_extractor = yake.KeywordExtractor(
                lan='en',
                n=2,  # Max 2-word phrases
                dedupLim=0.9,
                top=max_keywords * 2,  # Get more, filter later
                features=None
            )
            
            keywords = kw_extractor.extract_keywords(text_short)
            
            # Clean and filter keywords
            clean_keywords = []
            for kw, score in keywords:
                # Remove special characters
                kw_clean = re.sub(r'[^A-Za-z0-9\s]', '', kw).strip()
                
                # Skip if empty or in noise list
                if not kw_clean or kw_clean.lower() in noise_tokens:
                    continue
                
                # Skip single characters
                if len(kw_clean) < 2:
                    continue
                
                clean_keywords.append(kw_clean.title())
            
            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = []
            for kw in clean_keywords:
                if kw.lower() not in seen:
                    seen.add(kw.lower())
                    unique_keywords.append(kw)
            
            if unique_keywords:
                logger.info(f"✓ YAKE extracted {len(unique_keywords)} keywords")
                return unique_keywords[:max_keywords]
        
        except Exception as e:
            logger.warning(f"YAKE extraction failed: {e}")
    
    # Strategy 2: TF-IDF with smart filtering
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            max_features=max_keywords * 2,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.85
        )
        
        tfidf_matrix = vectorizer.fit_transform([text_short])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Sort by score
        word_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        
        # Filter and clean
        keywords = []
        for word, score in word_scores:
            word_clean = re.sub(r'[^A-Za-z0-9\s]', '', word).strip()
            
            if word_clean and word_clean.lower() not in noise_tokens and len(word_clean) > 2:
                keywords.append(word_clean.title())
        
        if keywords:
            logger.info(f"✓ TF-IDF extracted {len(keywords)} keywords")
            return keywords[:max_keywords]
    
    except Exception as e:
        logger.warning(f"TF-IDF extraction failed: {e}")
    
    # Strategy 3: Fallback - frequency analysis
    try:
        # Extract words (4+ characters)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text_short.lower())
        
        # Common stopwords
        stopwords = {
            'this', 'that', 'which', 'these', 'those', 'with', 'from',
            'have', 'will', 'could', 'should', 'would', 'there', 'their',
            'about', 'using', 'used', 'been', 'also', 'into', 'more',
            'some', 'such', 'when', 'what', 'where', 'were', 'them'
        }
        
        # Filter words
        filtered = [w for w in words if w not in stopwords and w not in noise_tokens]
        
        # Count frequency
        freq = Counter(filtered)
        
        # Extract top keywords (must appear at least 2 times)
        concepts = []
        for word, count in freq.most_common(max_keywords * 2):
            if count >= 2 and len(concepts) < max_keywords:
                concepts.append(word.capitalize())
        
        if concepts:
            logger.info(f"✓ Frequency analysis extracted {len(concepts)} keywords")
            return concepts
    
    except Exception as e:
        logger.warning(f"Frequency analysis failed: {e}")
    
    # Ultimate fallback
    logger.warning("All keyword extraction methods failed, using defaults")
    return ['General', 'Notes', 'Study Material']

def generate_table_of_contents_enhanced(sections, text):
    """
    Enhanced TOC generation WITHOUT confidence values.
    Combines both approaches: smart level detection, topic classification,
    page estimation, and clean previews.
    """
    from collections import defaultdict
    import re

    toc_entries = []
    by_level = defaultdict(list)
    by_topic = defaultdict(list)
    by_page = defaultdict(list)

    # Extract page numbers from text
    page_numbers = re.findall(r'\bpage\s+(\d+)\b', text.lower())
    max_page = max([int(p) for p in page_numbers], default=max(1, len(text)//3000))

    for idx, section in enumerate(sections, 1):
        title = section.get('title', f'Section {idx}')
        desc = section.get('desc', '') or ''

        # Determine hierarchical level
        if re.match(r'^\d+\.\d+\.\d+', title):
            level = 3
        elif re.match(r'^\d+\.\d+', title):
            level = 2
        elif any(word in title.lower() for word in ['subsection', 'subtopic', 'detail']):
            level = 2
        elif len(title.split()) > 8:
            level = 2
        else:
            level = 1

        # Estimate page number
        title_pos = text.lower().find(title.lower()[:20])
        page = min(max(1, title_pos // 3000), max_page) if title_pos >= 0 else 1

        # Classify topic
        try:
            topic, _ = classify_subject_advanced(f"{title} {desc}"[:600])
        except:
            topic = 'General'

        # Clean preview (remove extra whitespace and bullets)
        preview_text = re.sub(r'^[\s•]*', '', desc, flags=re.MULTILINE)
        preview_text = ' '.join(preview_text.split())
        preview = preview_text[:150] + '...' if len(preview_text) > 150 else preview_text

        entry = {
            'id': idx,
            'title': title,
            'level': level,
            'page': page,
            'topic': topic,
            'preview': preview
        }

        toc_entries.append(entry)
        by_level[level].append(entry)
        by_topic[topic].append(entry)
        by_page[str(page)].append(entry)

    # Summary statistics
    summary = {
        'total_sections': len(toc_entries),
        'main_sections': len([e for e in toc_entries if e['level'] == 1]),
        'main_topics': len(by_topic),
        'subtopics': sum(1 for e in toc_entries if e['level'] > 1),
        'pages_covered': len(by_page),
        'topics': list(by_topic.keys())
    }

    toc = {
        'entries': toc_entries,
        'summary': summary,
        'by_level': dict(by_level),
        'by_topic': dict(by_topic),
        'by_page': dict(by_page),
        'total_sections': len(toc_entries)
    }

    return toc

# ==================== API ROUTES ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check with feature detection"""
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        tesseract_status = f"v{tesseract_version}"
    except:
        tesseract_status = "not installed"
    
    features = [
        f"✓ Tesseract OCR {tesseract_status}",
        f"✓ Advanced text cleaning & normalization",
        f"✓ Intelligent section detection",
        f"✓ Smart summarization"
    ]
    
    if SPACY_AVAILABLE:
        features.append("✓ spaCy NER & NLP")
    if KEYBERT_AVAILABLE:
        features.append("✓ KeyBERT keyword extraction")
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        features.append("✓ Sentence-BERT embeddings")
    if YAKE_AVAILABLE:
        features.append("✓ YAKE keyword extraction")
    
    status = {
        'status': 'healthy',
        'message': 'NoteMap Enhanced - No OpenAI Required',
        'version': 'v5.0-enhanced',
        'features': features,
        'spacy_available': SPACY_AVAILABLE,
        'keybert_available': KEYBERT_AVAILABLE,
        'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
        'yake_available': YAKE_AVAILABLE,
        'tesseract_available': tesseract_status != "not installed",
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    """Upload and process file with enhanced NLP"""

    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        logger.info("=" * 70)
        logger.info("FILE UPLOAD REQUEST (Enhanced Processing)")
        logger.info("=" * 70)

        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        logger.info(f"Processing: {file.filename}")

        file_content = file.read()
        file_type = file.content_type
        logger.info(f"Type: {file_type}, Size: {len(file_content)} bytes")

        extracted_text = ""
        is_printed = False
        detection_confidence = 0.5
        ocr_method = "unknown"

        # -------------------------
        # FILE TYPE HANDLING
        # -------------------------
        if file_type == 'application/pdf':
            logger.info("Processing PDF...")

            try:
                # ✅ Hybrid extraction (direct text + OCR fallback)
                extracted_text, extraction_method = extract_pdf_with_ocr_fallback(file_content)
                is_printed = True
                detection_confidence = 0.95
                ocr_method = extraction_method

            except Exception as e:
                error_msg = str(e)
                logger.error(f"PDF processing completely failed: {error_msg}")
                return jsonify({
                    'success': False,
                    'error': 'PDF processing failed',
                    'details': error_msg,
                    'suggestion': 'Try converting to images or ensure PDF is not corrupted'
                }), 400

        elif file_type.startswith('image/'):
            logger.info("Processing image...")

            # Detect text type
            is_printed, detection_confidence = detect_text_type(file_content)

            # Perform OCR
            logger.info(f"OCR mode: {'PRINTED' if is_printed else 'HANDWRITTEN'}")
            extracted_text = perform_tesseract_ocr(file_content, is_printed=is_printed)
            ocr_method = f"Tesseract OCR ({'printed' if is_printed else 'handwritten'})"

        else:
            logger.error(f"Unsupported type: {file_type}")
            return jsonify({
                'success': False,
                'error': f'Unsupported file type: {file_type}'
            }), 400

        # Normalize extracted text
        extracted_text = normalize_extracted_text(extracted_text)

        # -------------------------
        # VALIDATION
        # -------------------------
        if not extracted_text or len(extracted_text.strip()) < 10:
            logger.error("Insufficient text extracted")
            return jsonify({
                'success': False,
                'error': 'No readable text found. Image may be too blurry or handwriting unclear.'
            }), 400

        logger.info(f"Extracted: {len(extracted_text)} characters")
        logger.info("Starting NLP analysis...")

        # -------------------------
        # ADVANCED NLP PIPELINE
        # -------------------------
        subject, subject_confidence = classify_subject_advanced(extracted_text)
        sections = smart_section_detection(extracted_text)
        concepts = extract_keywords_advanced(extracted_text)
        tableOfContents = generate_table_of_contents_enhanced(sections, extracted_text)

        logger.info(f"Subject: {subject} ({subject_confidence}% confidence)")
        logger.info(f"Sections: {len(sections)} detected")
        logger.info(f"Concepts: {len(concepts)} extracted")

        # Named Entity Recognition
        entities = []
        if SPACY_AVAILABLE:
            try:
                doc = nlp(extracted_text[:5000])
                entities = [
                    {
                        'text': ent.text,
                        'label': ent.label_,
                        'type': spacy.explain(ent.label_)
                    }
                    for ent in doc.ents
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'EVENT']
                ][:10]
                logger.info(f"Entities: {len(entities)} named entities")
            except Exception as e:
                logger.warning(f"NER failed: {e}")

        # -------------------------
        # TEXT METRICS
        # -------------------------
        word_count = len(extracted_text.split())
        char_count = len(extracted_text)
        sentence_count = len(sent_tokenize(extracted_text)) if extracted_text else 0
        avg_word_length = char_count / word_count if word_count > 0 else 0

        readability_score = None
        try:
            import textstat
            readability_score = {
                'flesch_reading_ease': round(textstat.flesch_reading_ease(extracted_text), 1),
                'grade_level': round(textstat.flesch_kincaid_grade(extracted_text), 1)
            }
        except:
            pass

        # -------------------------
        # RESPONSE
        # -------------------------
        
        response = {
            'success': True,
            'filename': file.filename,
            'subject': subject,
            'subject_confidence': subject_confidence,
            'preview': extracted_text[:300] + '...' if len(extracted_text) > 300 else extracted_text,
            'sections': sections,
            'concepts': concepts,
            'entities': entities,
            'tableOfContents': tableOfContents,
            'summary': tableOfContents.get('summary', ''),
            'totalSections': len(sections),
            'extractedText': extracted_text,
            'text': extracted_text,
            'date': datetime.now().strftime('%b %d'),
            'charCount': char_count,
            'wordCount': word_count,
            'sentenceCount': sentence_count,
            'avgWordLength': round(avg_word_length, 1),
            'readability': readability_score,
            'processingInfo': {
                'ocr_method': ocr_method,
                'text_type': 'printed' if is_printed else 'handwritten',
                'detection_confidence': round(detection_confidence * 100),
                'openai_used': False,
                'nlp_engine': 'spaCy + NLTK + Transformers',
                'features_used': {
                    'spacy': SPACY_AVAILABLE,
                    'keybert': KEYBERT_AVAILABLE,
                    'sentence_bert': SENTENCE_TRANSFORMERS_AVAILABLE,
                    'yake': YAKE_AVAILABLE
                }
            }
        }

        logger.info("✓ Processing complete!")
        logger.info(f"  Method: {ocr_method}")
        logger.info(f"  Subject: {subject} ({subject_confidence}%)")
        logger.info(f"  Sections: {len(sections)}")
        logger.info(f"  Concepts: {len(concepts)}")
        logger.info(f"  Entities: {len(entities)}")
        logger.info(f"  Words: {word_count}, Sentences: {sentence_count}")
        logger.info("=" * 70)

        return jsonify(response)

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error("ERROR:")
        logger.error(error_trace)
        return jsonify({
            'success': False,
            'error': f'Processing failed: {str(e)}',
            'details': str(e)
        }), 500
# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  NOTEMAP ENHANCED v5.0 - No OpenAI Required")
    print("="*70)
    
    # Check Tesseract
    try:
        tesseract_ver = pytesseract.get_tesseract_version()
        print(f"  ✓ Tesseract OCR v{tesseract_ver}")
    except Exception as e:
        print(f"  ✗ Tesseract: {e}")
        print(f"  Install from: https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)
    
    # Check NLP libraries
    print(f"\n  NLP Engines:")
    print(f"    {'✓' if SPACY_AVAILABLE else '✗'} spaCy (NER, POS tagging)")
    print(f"    {'✓' if KEYBERT_AVAILABLE else '✗'} KeyBERT (keyword extraction)")
    print(f"    {'✓' if SENTENCE_TRANSFORMERS_AVAILABLE else '✗'} Sentence-BERT (embeddings)")
    print(f"    {'✓' if YAKE_AVAILABLE else '✗'} YAKE (keyword extraction)")
    print(f"    ✓ NLTK (tokenization, stopwords)")
    
    if not SPACY_AVAILABLE:
        print(f"\n  ⚠️  Install spaCy for better results:")
        print(f"      pip install spacy")
        print(f"      python -m spacy download en_core_web_sm")
    
    if not KEYBERT_AVAILABLE:
        print(f"\n  ⚠️  Install KeyBERT for better keyword extraction:")
        print(f"      pip install keybert")
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print(f"\n  ⚠️  Install Sentence-Transformers for better summarization:")
        print(f"      pip install sentence-transformers")
    
    print(f"\n  Enhanced Features:")
    print(f"    • Advanced text cleaning & normalization")
    print(f"    • Intelligent printed/handwritten detection")
    print(f"    • Multi-strategy keyword extraction")
    print(f"    • Smart section detection (regex + ML)")
    print(f"    • Sentence-level summarization")
    print(f"    • Named entity recognition")
    print(f"    • Hierarchical TOC generation")
    print(f"    • Subject classification with confidence")
    print(f"    • Text quality metrics")
    
    print("\n" + "="*70)
    print("  Server: http://0.0.0.0:5000")
    print("  CORS: Enabled for localhost:3000")
    print("  Press CTRL+C to quit\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)