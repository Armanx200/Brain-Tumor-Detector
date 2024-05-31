```markdown
# 🧠 Brain Tumor Detection with Deep Learning

Welcome to the **Brain Tumor Detection** project! This repository contains a deep learning model that classifies brain images as either having a tumor or not. Dive into the power of convolutional neural networks (CNNs) and see how they can be used in medical imaging! 🚀

## 📁 Project Structure

```plaintext
brain-tumor-detection/
│
├── brain_tumor_dataset/
│   ├── no/         # Images without tumors
│   └── yes/        # Images with tumors
│
├── Brain-tumor-detector.py        # Model training and evaluation
└── Brain-tumor-probability.py     # Tumor probability prediction for a specific image
```

## 📜 Description

This project uses a CNN model to classify brain images into two categories:
- **No tumor** (stored in `brain_tumor_dataset/no`)
- **Tumor** (stored in `brain_tumor_dataset/yes`)

### 🛠️ Getting Started

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Armanx200/Brain-tumor-detector.git
    cd Brain-tumor-detector
    ```

2. **Install dependencies**:
    ```bash
    pip install tensorflow opencv-python scikit-learn
    ```

3. **Download the dataset**:
    - Ensure you have the brain tumor dataset in the correct structure as shown above.

### 🚀 Training the Model

Run the `Brain-tumor-detector.py` script to train the model:

```bash
python Brain-tumor-detector.py
```

This script:
- Loads and preprocesses the dataset.
- Splits the data into training and testing sets.
- Builds and trains a CNN model.
- Evaluates the model and displays the test accuracy.

### 🔍 Predicting Tumor Probability

To predict the probability of a specific image having a tumor, use the `Brain-tumor-probability.py` script:

1. Ensure the model is trained and saved as `brain_tumor_model.h5`.

2. Run the prediction script:

    ```bash
    python Brain-tumor-probability.py
    ```

    This script will:
    - Load the trained model.
    - Preprocess the specified image.
    - Output the probability of the image being a tumor.

## 📊 Model Performance

The model achieves impressive accuracy on the test set. See the script output for detailed performance metrics.

## 🛠️ Tools and Technologies

- **TensorFlow/Keras**: For building and training the neural network.
- **OpenCV**: For image processing.
- **Scikit-learn**: For data splitting and preprocessing.

## 📸 Sample Results

Below is an example of how the model predicts the probability of an image being a tumor:

![Sample Brain Image](https://github.com/Armanx200/Brain-tumor-detector/blob/main/Brain.jpg)

```
Probability of the image being a tumor: 95.44%
```

## 🤝 Contributing

Contributions are welcome! Please create a pull request or open an issue for any improvements or bug fixes.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Developed with ❤️ by [Armanx200](https://github.com/Armanx200)*
```
