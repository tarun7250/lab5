import cv2
import numpy as np

# Load frames from a video file
cap = cv2.VideoCapture('video.mp4')

# Initialize Lucas-Kanade algorithm
lk_params = dict(winSize=(3, 3), maxLevel=1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize block-matching algorithm
block_size = 16
search_range = 16
block_mask = np.ones((block_size, block_size), np.float32) / (block_size ** 2)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_blocks_x = video_width // block_size
num_blocks_y = video_height // block_size

while True



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
     if not ret:
        break
    next_frame = preprocess_frame(next_frame)

    # Compute Lucas-Kanade motion vectors
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    motion_vectors_lk, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, None, **lk_params)

    # Compute block-matching motion vectors
    motion_vectors_bm = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute deep network motion vectors
    motion_vectors_nn = estimate_motion(prev_frame, next_frame, motion_model)

    # Compute MSE between motion vectors and ground truth
    mse_lk += mse(motion_vectors_lk, motion_vectors)
    mse_bm += mse(motion_vectors_bm, motion_vectors)
    mse_nn += mse(motion_vectors_nn, motion_vectors)
    count += 1

    prev_frame = next_frame

cap.release()
mse_lk /= count
mse_bm /= count
mse_nn /= count

print('MSE Lucas-Kanade: %.4f' % mse_lk)
print('MSE Block-Matching: %.4f' % mse_bm)
print('MSE Deep Network: %.4f' % mse_nn)