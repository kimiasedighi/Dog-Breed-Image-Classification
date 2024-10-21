## Project Overview

This project is designed to classify dog breeds from images using a Convolutional Neural Network (CNN). It leverages the TensorFlow and Keras libraries to build, train, and evaluate the deep learning model. The dataset used consists of dog breed images, which are preprocessed and divided into training, validation, and test sets.

## Dataset

The dataset used is from Stanford’s ImageNet Dog Dataset. The script downloads and extracts the dataset from the following link:
- [ImageNet Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar)

The dataset is split into:
- **Training Set**: 80% of the dataset.
- **Validation Set**: 20% of the dataset.
- **Test Set**: 20% of the validation set.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

## Project Structure

1. **Import Libraries**: The required libraries are imported, including TensorFlow, Keras, NumPy, and Matplotlib.
   
2. **Dataset Download and Preprocessing**:
   - The dataset is downloaded, extracted, and loaded using `tensorflow.keras.preprocessing.image_dataset_from_directory`.
   - Image size is set to 200x200 pixels.
   - The dataset is cached and optimized for performance using `AUTOTUNE`.
   
3. **Model Architecture**:
   - The model is a Sequential CNN with three convolutional layers and max-pooling.
   - It includes data augmentation (random rotation) and a rescaling layer to normalize pixel values.
   - L2 regularization is applied to the convolutional layers to prevent overfitting.
   - The model ends with a fully connected layer followed by a dense layer with 120 units, representing the number of dog breeds.

4. **Model Compilation**:
   - The model is compiled using the Adam optimizer.
   - The loss function is Sparse Categorical Crossentropy, as the problem involves multi-class classification.
   - Accuracy is used as the evaluation metric.

5. **Model Training**:
   - The model is trained over 15 epochs with the training and validation datasets.

## Instructions to Run

1. Clone the repository or download the notebook.
2. Install the required dependencies listed in the `Requirements` section.
3. Run the Jupyter notebook (`hw3.ipynb`).
4. The script will download the dataset and start training the model.

## Model Performance

The model's performance is tracked using accuracy and loss metrics, evaluated on both the training and validation datasets after each epoch.

## Future Improvements

- Implement additional data augmentation techniques to improve generalization.
- Increase the complexity of the model by adding more layers or using pre-trained models for transfer learning.
- Experiment with different optimizers and learning rates to improve performance.

## Contact

If you have any questions or feedback, feel free to reach out to [Kimia Sedighi](mailto:your-email@example.com).
