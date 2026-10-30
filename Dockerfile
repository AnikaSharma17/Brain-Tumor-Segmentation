# CPU image for training / inference.
FROM python:3.11-slim

WORKDIR /app

# System deps for h5py / pillow wheels are already provided by slim + manylinux wheels.
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY scripts ./scripts

# The dataset is large and is mounted at runtime rather than baked into the image:
#   docker build -t brain-tumor-seg .
#   docker run --rm -v "$(pwd)/image_dataset:/app/image_dataset" \
#       -v "$(pwd)/outputs:/app/outputs" brain-tumor-seg
CMD ["python", "scripts/train.py"]
