import tensorflow as tf
from typing import List
import cv2
import os

# Define the vocabulary
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Convert characters to integers
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

# Convert integers back to characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


# Function to load video and preprocess frames
def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            continue  # Skip bad frames
        frame = tf.image.rgb_to_grayscale(frame)
        frame = frame[190:236, 80:220, :]  # Crop ROI
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No valid frames loaded from {path}")

    frames = tf.stack(frames)
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


# Function to load alignment transcript and convert to integer representation
def load_alignments(path: str) -> tf.Tensor:
    with open(path, 'r') as f:
        lines = f.readlines()

    tokens = ""
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[2] != 'sil':
            tokens += " " + parts[2]

    tokens = tokens.strip()
    chars = tf.strings.unicode_split(tokens, input_encoding='UTF-8')
    return char_to_num(chars)


# Function to load video and alignment using file path
def load_data(path: tf.Tensor):
    path = bytes.decode(path.numpy())  # Convert Tensor to string
    file_name = os.path.splitext(os.path.basename(path))[0]  # Extract file ID

    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')

    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments
