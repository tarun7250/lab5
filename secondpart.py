import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import FlowNetS

# Load pre-trained FlowNetS model
flownet_model = FlowNetS(weights='imagenet', include_top=False, input_shape=(384, 512, 6))

# Freeze all layers except for the last few layers
for layer in flownet_model.layers[:-10]:
    layer.trainable = False

# Add new layers for motion estimation
x = flownet_model.output
x = tf.keras.layers.Conv2D(filters=2, kernel_size=3, padding='same')(x)
motion_model = tf.keras.Model(inputs=flownet_model.input, outputs=x)

# Compile motion model with mean squared error loss
motion_model.compile(loss='mean_squared_error', optimizer='adam')

# Collect dataset of videos
videos = ['video1.mp4', 'video2.mp4', 'video3.mp4', 'video4.mp4', 'video5.mp4', 'video6.mp4', 'video7.mp4',
          'video8.mp4', 'video9.mp4', 'video10.mp4', 'video11.mp4', 'video12.mp4', 'video13.mp4', 'video14.mp4',
          'video15.mp4']

# Split dataset into training and testing sets
train_videos = videos[:12]
test_videos = videos[12:]

# Preprocess dataset
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (512, 384))
    return frame

train_frames = []
for video in train_videos:
    cap = cv2.VideoCapture(video)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess_frame(frame)
        train_frames.append(frame)
    cap.release()

test_frames = []
for video in test_videos:
    cap = cv2.VideoCapture(video)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess_frame(frame)
        test_frames.append(frame)
    cap.release()

# Train motion model
train_input = np.zeros((len(train_frames) - 1, 384, 512, 6), dtype=np.float32)
train_output = np.zeros((len(train_frames) - 1, 384, 512, 2), dtype=np.float32)
for i in range(len(train_frames) - 1):
    prev_frame = train_frames[i]
    next_frame = train_frames[i+1]
    train_input[i] = np.dstack((prev_frame, next_frame, np.zeros_like(prev_frame)))
    train_output[i] = motion_vectors

motion_model.fit(train_input, train_output, batch_size=4, epochs=10)

# Test motion model
def estimate_motion(frame1, frame2, model):
    input_data = np.dstack((frame1, frame2, np.zeros_like(frame1)))
    output_data = model.predict(np.array([input_data]))[0]
    return output_data

def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

mse_lk = 0.0
mse_bm = 0.0
mse_nn = 0.0
count = 0
for video in test_videos:
    cap = cv2.VideoCapture(video)
    ret, prev_frame = cap.read()
    prev_frame = preprocess_frame(prev_frame)
    while True:
        ret, next_frame = cap.read
