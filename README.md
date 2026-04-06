# Blood Group Detection (Image Processing) - Python API + Mini Website

This project provides a complete starter implementation for **blood grouping detection from an image** using Python image processing.

## Features
- Flask API endpoint for prediction (`/api/predict`)
- Mini web UI for image upload and result display (`/`)
- Image-processing based reaction detection for Anti-A, Anti-B, and Anti-D regions
- ABO + Rh output (`A+`, `O-`, `AB+`, etc.)
- Basic tests for classifier logic

## How it works
The uploaded test-card image is expected to contain 3 reagent regions in this order:
1. **Anti-A**
2. **Anti-B**
3. **Anti-D (Rh)**

The classifier:
1. converts image to grayscale,
2. splits the image into 3 vertical regions,
3. computes a texture/edge score per region to estimate agglutination,
4. maps positive/negative reactions to blood group.

> Note: This is an educational/demo implementation, not a medical device.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open: `http://127.0.0.1:5000`

## API usage

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "image=@/absolute/path/to/test-card.jpg"
```

Example response:

```json
{
  "blood_group": "A+",
  "reactions": {
    "anti_a": {"positive": true, "score": 0.63},
    "anti_b": {"positive": false, "score": 0.21},
    "anti_d": {"positive": true, "score": 0.71}
  },
  "confidence": 0.78
}
```

## Tests

```bash
python -m unittest -v
```
