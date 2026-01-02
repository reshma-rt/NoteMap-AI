import React, { useState } from "react";

const OCRUploader = ({ onNoteProcessed }) => {
  const [file, setFile] = useState(null);
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileUpload = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setImage(URL.createObjectURL(selectedFile));
      setResult(null);
      setError("");
    }
  };

  const handleProcessFile = async () => {
    if (!file) return alert("Please upload a file first!");
    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
        // Call the callback to add to processed notes
        if (onNoteProcessed) {
          onNoteProcessed(data);
        }
      } else {
        setError(data.error || "Processing failed");
      }
    } catch (err) {
      console.error("Upload Error:", err);
      setError("Failed to connect to server. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto text-center bg-gray-900 text-white rounded-2xl shadow-lg">
      <h2 className="text-2xl font-semibold mb-4">🧠 NoteMap OCR Processor</h2>
      <input
        type="file"
        accept="image/*,.pdf"
        onChange={handleFileUpload}
        className="mb-4 text-sm"
      />
      {image && (
        <img
          src={image}
          alt="Uploaded"
          className="mx-auto mb-4 max-h-64 rounded-lg border border-gray-700"
        />
      )}

      <button
        onClick={handleProcessFile}
        disabled={loading}
        className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-lg disabled:opacity-50"
      >
        {loading ? "Processing..." : "Process File"}
      </button>

      {error && (
        <div className="mt-4 bg-red-800 p-4 rounded-lg text-left">
          <h3 className="font-semibold text-red-400 mb-2">Error:</h3>
          <p className="text-gray-200">{error}</p>
        </div>
      )}

      {result && (
        <div className="mt-6 space-y-4">
          {/* Basic Info */}
          <div className="bg-gray-800 p-4 rounded-lg text-left">
            <h3 className="font-semibold text-indigo-400 mb-2">📄 File Info:</h3>
            <p><strong>Filename:</strong> {result.filename}</p>
            <p><strong>Subject:</strong> {result.subject}</p>
            <p><strong>Date:</strong> {result.date}</p>
            <p><strong>Characters:</strong> {result.charCount} | <strong>Words:</strong> {result.wordCount}</p>
          </div>

          {/* Processing Info */}
          <div className="bg-gray-800 p-4 rounded-lg text-left">
            <h3 className="font-semibold text-indigo-400 mb-2">⚙️ Processing Info:</h3>
            <p><strong>Method:</strong> {result.processingInfo.ocr_method}</p>
            <p><strong>Text Type:</strong> {result.processingInfo.text_type}</p>
            <p><strong>OpenAI Used:</strong> {result.processingInfo.openai_used ? "Yes" : "No"}</p>
          </div>

          {/* Preview */}
          <div className="bg-gray-800 p-4 rounded-lg text-left">
            <h3 className="font-semibold text-indigo-400 mb-2">👀 Preview:</h3>
            <p className="text-gray-200">{result.preview}</p>
          </div>

          {/* Concepts */}
          <div className="bg-gray-800 p-4 rounded-lg text-left">
            <h3 className="font-semibold text-indigo-400 mb-2">💡 Key Concepts:</h3>
            <div className="flex flex-wrap gap-2">
              {result.concepts.map((concept, idx) => (
                <span key={idx} className="bg-indigo-600 px-2 py-1 rounded text-sm">
                  {concept}
                </span>
              ))}
            </div>
          </div>

          {/* Table of Contents Summary */}
          {result.summary && (
            <div className="bg-gray-800 p-4 rounded-lg text-left">
              <h3 className="font-semibold text-indigo-400 mb-2">📚 Table of Contents Summary:</h3>
              <p><strong>Total Sections:</strong> {result.summary.total_sections}</p>
              <p><strong>Main Topics:</strong> {result.summary.main_topics}</p>
              <p><strong>Subtopics:</strong> {result.summary.subtopics}</p>
              <p><strong>Pages Covered:</strong> {result.summary.pages_covered}</p>
              <p><strong>Average Confidence:</strong> {result.summary.avg_confidence}%</p>
            </div>
          )}

          {/* Sections */}
          <div className="bg-gray-800 p-4 rounded-lg text-left">
            <h3 className="font-semibold text-indigo-400 mb-2">📑 Sections ({result.totalSections}):</h3>
            <div className="space-y-2">
              {result.sections.map((section, idx) => (
                <div key={idx} className="border-l-4 border-indigo-500 pl-4">
                  <h4 className="font-medium text-indigo-300">{section.title}</h4>
                  <p className="text-gray-300 text-sm">{section.desc}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Full Text */}
          <div className="bg-gray-800 p-4 rounded-lg text-left">
            <h3 className="font-semibold text-indigo-400 mb-2">📖 Full Extracted Text:</h3>
            <pre className="whitespace-pre-wrap text-gray-200 text-sm max-h-96 overflow-y-auto">
              {result.extractedText}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};

export default OCRUploader;
