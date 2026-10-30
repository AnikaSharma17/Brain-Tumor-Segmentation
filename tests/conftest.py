import os

# Quiet TensorFlow logs during tests.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
