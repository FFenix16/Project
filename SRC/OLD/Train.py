import os
import numpy as np
import tensorflow as tf
from keras.applications import VGG16
from keras.preprocessing import image
import xml.etree.ElementTree as ET



# Set the target size for image resizing.
target_size = (480, 480)

# Create a model.
base_model = VGG16(input_shape=(target_size[0], target_size[1], 3), include_top=False)
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Directory paths
data_dir = 'img\\Ai'  # Directory containing the 'test', 'train', and 'valid' folders
test_dir = os.path.join(data_dir, 'test')
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')

# Load the images and annotations.
def load_data_from_directory(image_dir, annotation_dir):
    images = []
    labels = []
    
    # Get a list of image filenames
    image_filenames = os.listdir(image_dir)
    
    # Iterate over the image filenames
    for filename in image_filenames:
        # Load the image
        img = image.load_img(os.path.join(image_dir, filename), target_size=target_size)
        img_array = image.img_to_array(img)
        images.append(img_array)
        
        # Load the corresponding annotation
        annotation_filename = os.path.splitext(filename)[0] + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_filename)
        
        # Parse the XML annotation file
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Extract the label (assuming it's stored as a child tag named 'label')
        label = root.find('label').text
        labels.append(label)
    
    return images, labels

# Load test data
test_images, test_labels = load_data_from_directory(os.path.join(test_dir, 'images'), os.path.join(test_dir, 'annotations'))

# Load train data
train_images, train_labels = load_data_from_directory(os.path.join(train_dir, 'images'), os.path.join(train_dir, 'annotations'))

# Load valid data
valid_images, valid_labels = load_data_from_directory(os.path.join(valid_dir, 'images'), os.path.join(valid_dir, 'annotations'))

# Convert the lists to numpy arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)
train_images = np.array(train_images)
train_labels = np.array(train_labels)
valid_images = np.array(valid_images)
valid_labels = np.array(valid_labels)

# Normalize the pixel values
test_images = test_images / 255.0
train_images = train_images / 255.0
valid_images = valid_images / 255.0

# Convert labels to one-hot encoding
num_classes = 10
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
valid_labels = tf.keras.utils.to_categorical(valid_labels, num_classes)

# Train the model.
model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(valid_images, valid_labels)
)

# Save the model.
model.save('model.h5')
