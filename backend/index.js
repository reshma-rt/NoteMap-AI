const express = require('express');
const fileUpload = require('express-fileupload');
const cors = require('cors');
const vision = require('@google-cloud/vision');

const app = express();
app.use(cors());
app.use(fileUpload());

const client = new vision.ImageAnnotatorClient(); // uses GOOGLE_APPLICATION_CREDENTIALS

app.post('/api/ocr', async (req, res) => {
  if (!req.files || !req.files.file) return res.status(400).send('No file uploaded');

  try {
    const file = req.files.file;
    const [result] = await client.textDetection(file.data);
    const detections = result.textAnnotations;
    res.json({ text: detections[0] ? detections[0].description : '' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'OCR failed' });
  }
});

app.listen(8000, () => console.log('OCR backend running on port 8000'));
