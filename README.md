# Image Caption Generator

A Deep Learning model that automatically generates descriptive captions for images using **CNN (InceptionV3)** and **LSTM**. This project is trained on the **Flickr8k dataset sourced from Kaggle** and combines Computer Vision with Natural Language Processing (NLP) to generate human-like text.

## Features
* **Feature Extraction**: Uses a pre-trained **InceptionV3** model (Transfer Learning) to extract features from images.
* **Sequence Processing**: Uses **LSTM (Long Short-Term Memory)** layers to handle text sequences.
* **Architecture**: Encoder-Decoder model that merges image vectors with text embeddings.
* **Greedy Search**: Generates captions by predicting the most likely next word.
* **Dataset**: Utilizes the popular **Flickr8k dataset** from Kaggle.

## Tech Stack
* **Language**: Python 3.10+
* **Deep Learning**: TensorFlow / Keras
* **Image Processing**: Pillow (PIL), NumPy
* **Data Handling**: Pandas, tqdm
* **Evaluation**: NLTK (BLEU Score)

## Dataset
This project uses the **Flickr8k Dataset**. You must download it from Kaggle to run the code.

* **Dataset Name**: Flickr8k
* **Source**: [Kaggle - Flickr8k (adityajn105)](https://www.kaggle.com/datasets/adityajn105/flickr8k)
* **Content**: ~8,000 images with 5 captions per image.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/image-caption-generator.git](https://github.com/your-username/image-caption-generator.git)
    cd image-caption-generator
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install numpy matplotlib pandas seaborn tensorflow scikit-learn pillow tqdm nltk
    ```

## Project Structure
Ensure your folder structure matches the configuration in `main.py`.

  ```text
  Image-Caption_Generator/
  │
  ├── kaggle/
  │   ├── Images/           <-- Extract the 8091 images from Kaggle here
  │   └── captions.txt      <-- Place the captions.txt file here
  │
  ├── main.py               <-- Main training and prediction script
  └── README.md
  ```

## Usage
Download the Data: Go to the Flickr8k Kaggle Page, download the archive, and extract it into the kaggle/ folder.

### Run the Script:

  ```bash
  python main.py
  ```

## Process:

* **Feature Extraction:** The script will first pass all images through InceptionV3 to extract features (this happens once).
* **Training:** The model trains for 15 epochs.
* **Prediction:** The script will output generated captions for 5 random test images.
