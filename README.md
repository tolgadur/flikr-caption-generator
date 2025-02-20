# Flickr Caption Generator

A deep learning-based image captioning system that generates natural language descriptions for images using the Flickr dataset. The project implements a transformer-based architecture for image understanding and caption generation, with a FastAPI-based web service for easy deployment. The model weights were not included in the repository due to size constraints. You can easy retrain it though.

## 🌟 Features

- Transformer-based image captioning model
- FastAPI web service for real-time caption generation
- Docker support for easy deployment
- Weights & Biases integration for experiment tracking
- Comprehensive evaluation metrics
- Training and inference pipelines

## 🛠 Tech Stack

- Python 3.x
- PyTorch 2.1.0
- Transformers
- FastAPI
- Docker
- Weights & Biases

## 📋 Prerequisites

- Python 3.x
- CUDA-compatible GPU (for training)
- Docker (optional, for containerized deployment)


## 📁 Project Structure

```
flikr-caption-generator/
├── src/
│   ├── api.py           # FastAPI web service
│   ├── main.py          # Application entry point
│   ├── model.py         # Neural network architecture
│   ├── trainer.py       # Training logic
│   ├── decoder.py       # Caption generation decoder
│   ├── dataset.py       # Data loading and processing
│   ├── evals.py         # Evaluation metrics
│   ├── utils.py         # Utility functions
│   └── config.py        # Configuration settings
├── models/              # Saved model checkpoints
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
└── README.md
```

## 🎯 Usage

### Training

To train the model:

```bash
python src/trainer.py
```

The training progress can be monitored through Weights & Biases dashboard.

### Inference

The model serves predictions through a FastAPI web service. After starting the server:

```bash
python src/main.py
```

Access the API at `http://localhost:8000`.


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

For questions and feedback, please open an issue in the GitHub repository.
