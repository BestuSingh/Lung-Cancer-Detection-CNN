import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import os

# Define dataset paths
train_folder = r'C:\Users\Hp VICTUS\Downloads\archive (3)\nail_disease_dataset\train'
test_folder = r'C:\Users\Hp VICTUS\Downloads\archive (3)\nail_disease_dataset\test'

# Image size and batch size
IMAGE_SIZE = (224, 224)
batch_size = 8
num_classes = 4

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_folder, target_size=IMAGE_SIZE, batch_size=batch_size, class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_folder, target_size=IMAGE_SIZE, batch_size=batch_size, class_mode='categorical'
)

# Get class labels
class_labels = list(train_generator.class_indices.keys())

# Load VGG16 without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ModelCheckpoint('vgg16_lung_model.h5', monitor='val_loss', save_best_only=True)
]

# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=30,
    callbacks=callbacks
)

# Save final model
model.save('enhanced_model_vgg16.h5')

# Evaluation
y_true = []
y_pred = []

validation_generator.reset()
for _ in range(len(validation_generator)):
    x_batch, y_batch = next(validation_generator)
    preds = model.predict(x_batch)
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Prediction function for frontend
def predict_image(image_path, model_path='enhanced_model_vgg16.h5'):
    model = load_model(model_path)
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]
    return predicted_label
