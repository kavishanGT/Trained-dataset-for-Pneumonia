import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Resizing, Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Data paths (replace with your paths)
data_dir = "training/images"
train_data_dir = os.path.join(data_dir, "train")  # Assuming train directory is created after split
val_data_dir = os.path.join(data_dir, "val")  # Assuming val directory is created after split
test_data_dir = os.path.join(data_dir, "test")  # Assuming test directory is created after split

# Define image dimensions
img_width, img_height = 150, 150

# Function to split data into training, validation, and test sets
def split_data(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    categories = ['NORMAL','PNEUMONIA' ]
    for catogary in categories:
        catogary_dir = os.path.join(data_dir, catogary)
        all_filenames = os.listdir(catogary_dir)
    
  # Get all image filenames
    

  # Shuffle filenames for randomness
        random.shuffle(all_filenames)

  # Calculate number of images in each split
        train_size = int(len(all_filenames) * train_ratio)
        val_size = int(len(all_filenames) * val_ratio)
        test_size = len(all_filenames) - train_size - val_size
        print(train_size)

  # Split data into lists
        train_data = all_filenames[:train_size]
        val_data = all_filenames[train_size:train_size + val_size]
        test_data = all_filenames[train_size + val_size:]

        # Create directories for each category in train, val, and test dirs
        os.makedirs(os.path.join(train_data_dir, catogary), exist_ok=True)
        os.makedirs(os.path.join(val_data_dir, catogary), exist_ok=True)
        os.makedirs(os.path.join(test_data_dir, catogary), exist_ok=True)

        # Move images to respective directories
        for filename in train_data:
            src = os.path.join(catogary_dir, filename)
            dst = os.path.join(train_data_dir, catogary,filename)
            shutil.move(src, dst)  # Replace with shutil.copy() to copy if needed

        for filename in val_data:
            src = os.path.join(catogary_dir, filename)
            dst = os.path.join(val_data_dir,catogary, filename)
            shutil.move(src, dst)

        for filename in test_data:
            src = os.path.join(catogary_dir, filename)
            dst = os.path.join(test_data_dir,catogary, filename)
            shutil.move(src, dst)


# Split data (call the function)
split_data(data_dir)

# Data preprocessing using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=lambda x: tf.image.resize(x, (img_width, img_height)))
val_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=lambda x: tf.image.resize(x, (img_width, img_height)))

# Load training and validation data (using flow_from_directory)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'  # Binary classification (normal/pneumonia)
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)



# Define the CNN model
#n_classes = 3
model = Sequential([
    Resizing(img_width, img_height),
    Rescaling(1.0/255),
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=30,
    callbacks=[early_stopping]
)


# Save the model with the correct file extension
model_version = "1.0"  # You can change this to your preferred versioning
model.save(f"../t_models/cnn_model_v{model_version}.h5")