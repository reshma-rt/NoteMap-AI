const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const vision = require("@google-cloud/vision");
const cors = require("cors");

const app = express();
const upload = multer({ dest: "uploads/" });
const PORT = 8000;

app.use(cors());

// 🔐 Make sure this file path is correct
// You must have your service account key JSON downloaded from Google Cloud Console
// and set this environment variable in your terminal or VSCode before starting server:
// export GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"

const client = new vision.ImageAnnotatorClient();

// 🧠 OCR route using Google Cloud Vision
app.post("/extract-text", upload.single("image"), async (req, res) => {
  try {
    const filePath = path.resolve(req.file.path);

    // Call Google Vision API
    const [result] = await client.textDetection(filePath);
    const detections = result.textAnnotations;

    const extractedText = detections.length ? detections[0].description : "";

    console.log("✅ Extracted Text:\n", extractedText);

    // Clean up uploaded file
    fs.unlinkSync(filePath);

    res.json({ extractedText });
  } catch (err) {
    console.error("❌ Google OCR Error:", err);
    res.status(500).json({ error: "OCR failed", details: err.message });
  }
});

app.listen(PORT, () => console.log(`🚀 OCR server running at http://localhost:${PORT}`));
