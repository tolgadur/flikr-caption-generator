# Flickr Caption Generator

A deep learning-based image captioning system that generates natural language descriptions for images using the Flickr dataset. I used the CLIP model from Huggingface for image and text encodings and implemented a decoder with self-attention only for caption generation based on them. I took inspiration from the Pixtral 12B paper and instead of using cross-attention, I feed in the CLS token of my image encoding as the start token for a more streamlined architecture.

For inference, I sample captions at different temperatures and let CLIP select the best caption. The project implements a transformer-based architecture for image understanding and caption generation, with a FastAPI-based web service for easy deployment. The model weights were not included in the repository due to size constraints, but you can easily retrain it using the training script.

## 📁 Project Structure

```text
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

## 🐳 Docker Deployment

Update the registry in `docker-compose.yml`:

```yaml
image: docker.io/yourusername/flikr-caption-generator:backend
image: docker.io/yourusername/flikr-caption-generator:frontend
```

Build and push:

```bash
# Build both services
docker compose build

# Push to registry
docker compose push
```

Run on your server:

```bash
docker compose pull && docker compose up -d
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

For questions and feedback, please open an issue in the GitHub repository.
