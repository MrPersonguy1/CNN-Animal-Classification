🐾 CNN Animal Image Classification
This project uses a Convolutional Neural Network (CNN) to identify which animal is in a photo. It can classify images into 10 different animal categories, making it a fun and practical computer vision project using deep learning.

📚 What You’ll Learn
How to load and process image data

How to build a CNN model in PyTorch

How to train a model to recognize different animals

How to test the model and measure its accuracy

How to visualize both the training process and predictions

🛠️ Technologies Used
Python 🐍

PyTorch (for building the CNN)

torchvision (for datasets and image transforms)

NumPy and pandas (for working with data)

Matplotlib (for plotting results)

🧠 How It Works
Images are loaded and transformed to the right size and format.

A CNN model is built with multiple convolution and pooling layers.

The model is trained over 25 epochs using a dataset of animal images.

Accuracy and loss are plotted after training to show performance.

The trained model is tested on new images to see how well it recognizes animals.

Predictions are compared to true labels, and results are visualized.

🚀 How to Run
Install Python and PyTorch.

Install needed packages:

bash
Copy
Edit
pip install torch torchvision matplotlib pandas numpy
Load your dataset into the right directory (./data/animals or similar).

Run the script in a Python file or a Jupyter Notebook.

The training will begin, and charts will appear showing the model’s progress.

📁 File Overview
CNN_Animal_Classification.py: Contains all code for building, training, testing, and evaluating the CNN model on an animal image dataset.
