#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk  # Add this line to import Image module
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Global variables
dataset_loaded = False
dataset_path = ""
train_ds = None
val_ds = None
test_ds = None
model = None
class_names = None

def load_data():
    global dataset_loaded
    global dataset_path
    
    dataset_path = filedialog.askdirectory()
    
    if dataset_path:
        dataset_loaded = True
        message_text.insert(tk.END, "Data loaded successfully from folder: {}\n".format(dataset_path))
    else:
        message_text.insert(tk.END, "No folder selected\n")
        
    message_text.see(tk.END)

"""def process_data(dataset_path, img_height=32, img_width=32, batch_size=20):
    global train_ds
    global val_ds
    global test_ds
    global class_names
    
    if not dataset_path:
        message_text.insert(tk.END, "No dataset loaded. Please load the dataset first.\n")
        return None, None, None
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path + "/train",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path + "/validation",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path + "/test",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names

    message_text.insert(tk.END, "Data processed successfully.\n")
    message_text.see(tk.END)

    return train_ds, val_ds, test_ds"""

def process_data(dataset_path, img_height=32, img_width=32, batch_size=20):
    global train_ds
    global val_ds
    global test_ds
    global class_names
    
    if not dataset_path:
        message_text.insert(tk.END, "No dataset loaded. Please load the dataset first.\n")
        return None, None, None
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path + "/train",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path + "/validation",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path + "/test",
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names

    # Print the message in the GUI interface
    train_files = len(train_ds.file_paths)
    val_files = len(val_ds.file_paths)
    test_files = len(test_ds.file_paths)
    
    message_text.insert(tk.END, f"Found {train_files} files belonging to {len(class_names)} classes in training dataset.\n")
    message_text.insert(tk.END, f"Found {val_files} files belonging to {len(class_names)} classes in validation dataset.\n")
    message_text.insert(tk.END, f"Found {test_files} files belonging to {len(class_names)} classes in test dataset.\n")
    
    message_text.see(tk.END)

    return train_ds, val_ds, test_ds


def build_and_train_model():
    global train_ds
    global val_ds
    global model
    
    if not dataset_loaded:
        message_text.insert(tk.END, "No dataset loaded. Please load the dataset first.\n")
        return
    
    if train_ds is None or val_ds is None:
        message_text.insert(tk.END, "Dataset not processed. Please process the dataset first.\n")
        return
    
    model = tf.keras.Sequential(
        [
         tf.keras.layers.Rescaling(1./255),
         tf.keras.layers.Conv2D(32, 3, activation="relu"),
         tf.keras.layers.MaxPooling2D(),
         tf.keras.layers.Conv2D(32, 3, activation="relu"),
         tf.keras.layers.MaxPooling2D(),
         tf.keras.layers.Conv2D(32, 3, activation="relu"),
         tf.keras.layers.MaxPooling2D(),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(128, activation="sigmoid"),
         tf.keras.layers.Dense(3)
        ]
    )
    
    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
    )
    
    message_text.insert(tk.END, "Model trained successfully.\n")
    message_text.see(tk.END)

def predict_image():
    global test_ds
    global model
    global class_names
    
    if test_ds is None:
        message_text.insert(tk.END, "Test dataset not loaded. Please load the dataset first.\n")
        return
    
    if model is None:
        message_text.insert(tk.END, "Model not built and trained. Please build and train the model first.\n")
        return
    
    if class_names is None:
        message_text.insert(tk.END, "Class names not defined. Please process the dataset first.\n")
        return
    
    # Ask user to select an image file
    image_path = filedialog.askopenfilename()
    if not image_path:
        message_text.insert(tk.END, "No image selected\n")
        return
    
    # Display the image in a new window
    image_window = tk.Toplevel(root)
    image_window.title("Selected Image")
    image_window.geometry("400x400")
    
    # Load and display the image
    img = Image.open(image_path)
    img = img.resize((400, 400), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(image_window, image=photo)
    label.image = photo  # Keep a reference to avoid garbage collection
    label.pack()
    
    # Predict on the selected image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_names[predicted_class_index]

    message_text.insert(tk.END, "Prediction: {}\n".format(predicted_class_label))
    message_text.see(tk.END)

def exit_application():
    root.destroy()


# GUI setup
root = tk.Tk()
root.title("AI-DRIVEN DYNAMIC PRICING FOR TRAVEL SERVICE ")
root.configure(bg="#1E8449") 

# Title Frame
title_frame = tk.Frame(root, bg="#AF7AC5")
title_frame.pack(fill=tk.X, padx=20, pady=10)

title_label = tk.Label(title_frame, text="Cancer Prediction", font=("Helvetica", 18), bg="#AF7AC5", fg="white")
title_label.pack(pady=10)

# Button Frame
button_frame = tk.Frame(root, bg="#AF7AC5")
button_frame.pack(side=tk.LEFT, padx=20, pady=19)

load_button = tk.Button(button_frame, text="Load Data", command=load_data, bg="#AF7AC5", fg="white", height=2)
load_button.pack(side=tk.TOP, padx=(0, 10), pady=10, fill=tk.X)

process_button = tk.Button(button_frame, text="Process Data", command=lambda: process_data(dataset_path), bg="#AF7AC5", fg="white", height=2)
process_button.pack(side=tk.TOP, padx=(0, 10), pady=10, fill=tk.X)

build_train_button = tk.Button(button_frame, text="Build and Train Model", command=build_and_train_model, bg="#AF7AC5", fg="white", height=2)
build_train_button.pack(side=tk.TOP, padx=(0, 10), pady=10, fill=tk.X)

predict_button = tk.Button(button_frame, text="Predict", command=predict_image, bg="#AF7AC5", fg="white", height=2)
predict_button.pack(side=tk.TOP, padx=(0, 10), pady=10, fill=tk.X)

exit_button = tk.Button(button_frame, text="Exit", command=exit_application, bg="#AF7AC5", fg="white", height=3)
exit_button.pack(side=tk.TOP, padx=(0, 10), pady=(10, 10), fill=tk.X)

# Message Frame
message_frame = tk.Frame(root, bg="#F4D03F")
message_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

message_label = tk.Label(message_frame, text="Messages", font=("Helvetica", 14), bg="#F4D03F", fg="black")
message_label.pack(pady=5)

message_text = tk.Text(message_frame, bg="white", wrap=tk.WORD)
message_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=15)

root.mainloop()


# In[ ]:




