import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Load data from CSV file
xtrain_url = "https://raw.githubusercontent.com/smomtahe/ensemble-learning-absorption/main/phantom.csv"
data = pd.read_csv(url)

# Select features and target variables
features = ['reflectance1', 'reflectance2']
targets = ['absorption1', 'absorption2', 'side']

# Group the data into 8 equal parts and calculate the average
grouped_data = data.groupby(np.arange(len(data)) // 16).mean()

X = grouped_data[features].values
y = grouped_data[targets].values

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Load and preprocess the y train image
image_path = "https://raw.githubusercontent.com/smomtahe/RNN-Image-Reconstruction/main/image.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is not None:
    image_scaled = scaler.fit_transform(image.reshape(-1, 1))
else:
    print("Failed to load the image.")

# Prepare the input data for transfer learning
X_train = X_scaled.reshape(-1, 1, 2, 1)
X_image = image_scaled.reshape(-1, 1, 64, 64)

# Load the VGG16 model with pre-trained weights (excluding the top fully connected layers)
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Freeze the layers so they are not retrained during the transfer learning process
for layer in vgg16.layers:
    layer.trainable = False

# Extract features from the reflectance data and the image data
X_features = vgg16.predict(X_image)
X_features = X_features.reshape(X_features.shape[0], -1)

# Concatenate the features from the reflectance data and the image data
X_final = np.concatenate([X_train, X_features], axis=2)

# Build the prediction model
model = Model(inputs=vgg16.input, outputs=Dense(128, activation='relu')(X_final))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(3))  # Output layer with 3 units for absorption1, absorption2, and side

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
if image is not None:
    model.fit(X_image, y, epochs=10, batch_size=32)
else:
    print("Skipping model training due to image loading failure.")

# Load and preprocess the test set
url_test = "https://raw.githubusercontent.com/maryammomtahen/Testset/main/P14_S4_U_post.csv"
datatest = pd.read_csv(url_test)

X_test = datatest[features].values
X_test_scaled = scaler.transform(X_test)
X_test_scaled = X_test_scaled.reshape(-1, 1, 2, 1)

# Predict on the test dataset
y_pred = model.predict(X_test_scaled)

# Separate the predicted values for absorption1, absorption2, and side
absorption1_pred = y_pred[:, 0]  # Predicted absorption1 values
absorption2_pred = y_pred[:, 1]  # Predicted absorption2 values
side_pred = y_pred[:, 2]  # Predicted side values

# Scale absorption1_pred and absorption2_pred between 0 and 1
absorption1_pred_scaled = (absorption1_pred - np.min(absorption1_pred)) / (np.max(absorption1_pred) - np.min(absorption1_pred))
absorption2_pred_scaled = (absorption2_pred - np.min(absorption2_pred)) / (np.max(absorption2_pred) - np.min(absorption2_pred))

# Calculate (absorption2_pred + absorption1_pred)^2
result = (absorption2_pred_scaled + absorption1_pred_scaled) ** 2

# Find the most frequent side value
most_frequent_side = np.round(np.mean(side_pred)).astype(int)

# Map the most frequent side value to the corresponding label
side_label = ""
if most_frequent_side == 0:
    side_label = "Left"
elif most_frequent_side == 1:
    side_label = "Center"
elif most_frequent_side == 2:
    side_label = "Right"

print("absorption1_pred_scaled=")  # Predicted values for absorption1
print('[', ','.join(str(value) for value in absorption1_pred_scaled), '];')
print("absorption2_pred_scaled=")  # Predicted values for absorption2
print('[', ','.join(str(value) for value in absorption2_pred_scaled), '];')
print("most_frequent_side_label=")  # Most frequent predicted side label
print(side_label)

# Scale absorption1_pred and absorption2_pred between 0 and 1
absorption1_pred_scaled = (absorption1_pred - np.min(absorption1_pred)) / (np.max(absorption1_pred) - np.min(absorption1_pred))
absorption2_pred_scaled = (absorption2_pred - np.min(absorption2_pred)) / (np.max(absorption2_pred) - np.min(absorption2_pred))

# Plot absorption1_pred_scaled, absorption2_pred_scaled, and (absorption1_pred_scaled + absorption2_pred_scaled)^2
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(absorption1_pred_scaled, label='absorption1_pred_scaled')
plt.plot(absorption2_pred_scaled, label='absorption2_pred_scaled')
plt.xlabel("Index")
plt.ylabel("Scaled Predictions")
plt.title("Scaled Absorption Predictions")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(result)
plt.xlabel("Index")
plt.ylabel("(absorption1_pred_scaled + absorption2_pred_scaled)^2")
plt.title("Squared Sum of Scaled Absorption Predictions")

plt.tight_layout()
plt.show()
