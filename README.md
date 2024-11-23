
# Landmark Detection System

## Overview
This project is a **Landmark Detection System** built using **FastAPI** for the backend and **React** for the frontend. It leverages a **deep learning model** (EfficientNet) to predict landmarks in images. The application also integrates monitoring tools like **Prometheus** and **Grafana** and is fully containerized with Docker for seamless deployment.

---

## Features
- **FastAPI Backend**
  - Predict landmarks from uploaded images.
  - Health check endpoint.
  - Prometheus metrics for monitoring requests and latency.

- **React Frontend**
  - User-friendly UI for uploading images and viewing results.
  - Real-time image preview before prediction.
  - Clear error messages for invalid inputs.

- **Deep Learning Model**
  - TensorFlow-based EfficientNet model.
  - Configurable parameters for training and prediction.
  - Outputs top-K predictions with confidence scores.

- **Monitoring**
  - Prometheus for collecting API metrics.
  - Grafana for visualizing API performance.

- **Dockerized Deployment**
  - Backend and frontend services containerized.
  - Docker Compose for multi-service orchestration.

---

## Tech Stack
- **Backend**: Python, FastAPI, TensorFlow
- **Frontend**: React, TailwindCSS
- **Monitoring**: Prometheus, Grafana
- **Deployment**: Docker, Docker Compose

---

## Installation

### Prerequisites
- [Docker](https://www.docker.com/get-started) and Docker Compose
- Python 3.9+ (for local development)
- Node.js (for frontend development)

### Clone the Repository
```bash
git clone https://github.com/Nizar04/Landmark-Detection.git
cd landmark-detection
```

### Backend Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the FastAPI server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

### Frontend Setup
1. Navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm start
   ```

---

## Dockerized Deployment

### Build and Run Services
Use Docker Compose to start all services, including the backend, Prometheus, and Grafana:
```bash
docker-compose up --build
```

### Access Services
- **API**: [http://localhost:8000](http://localhost:8000)
- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **Prometheus**: [http://localhost:9090](http://localhost:9090)
- **Grafana**: [http://localhost:3000](http://localhost:3000)

---

## Usage

### Predict Landmarks
1. Open the frontend at [http://localhost:3000](http://localhost:3000).
2. Upload an image file.
3. Click "Detect Landmarks" to get predictions.

### Health Check
Ensure the backend is running by visiting:
```bash
http://localhost:8000/health
```

### Prometheus Metrics
API metrics are available at:
```bash
http://localhost:8000/metrics
```

---

## Configuration

### Backend
Configuration options are available in `app/config.py`, including:
- Model parameters
- Data augmentation settings
- Training hyperparameters

### Docker
Update environment variables in `docker-compose.yml` for paths like:
- `MODEL_PATH`
- `LABEL_ENCODER_PATH`

---

## Testing

### Run Backend Tests
```bash
pytest
```

### Test Frontend Components
Use [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/) or similar tools.

---

## File Structure

```
landmark-detection/
├── app/
│   ├── main.py             # FastAPI application
│   ├── models.py           # API response models
│   ├── inference.py        # Model inference logic
│   ├── config.py           # Configuration settings
│   ├── data_loader.py      # Data preprocessing
│   ├── model.py            # TensorFlow model definition
│   └── tests/              # Unit tests
├── frontend/               # React frontend application
├── Dockerfile              # Dockerfile for backend
├── docker-compose.yml      # Multi-service orchestration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Contributors
- **Nizar El Mouaquit(mailto:nizarelmouaquit@protonmail.com)**

Feel free to reach out for any questions or issues, or create a GitHub issue!

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
