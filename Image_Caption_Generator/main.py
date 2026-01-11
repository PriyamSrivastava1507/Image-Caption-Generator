import os
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import re
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

images_directory = r'kaggle\Images'
captions_path = r'kaggle\captions.txt'

if not os.path.exists(images_directory):
    print(f"CRITICAL ERROR: Images folder not found at: {images_directory}")
    exit()
if not os.path.exists(captions_path):
    print(f"CRITICAL ERROR: Captions file not found at: {captions_path}")
    exit()


def load_captions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        captions = f.readlines()
    captions = [caption.lower() for caption in captions[1:]]
    return captions


def tokenize_captions(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer


captions = load_captions(captions_path)


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


cleaned_captions = [clean_text(caption.split(',')[1]) for caption in captions]

captions_IDs = []
for i in range(len(cleaned_captions)):
    item = captions[i].split(',')[0] + '\t' + 'start ' + cleaned_captions[i] + ' end\n'
    captions_IDs.append(item)

max_caption_length = max(len(caption.split()) for caption in cleaned_captions) + 1

tokenizer = tokenize_captions(cleaned_captions)
vocab_size = len(tokenizer.word_index) + 1

all_image_ids = os.listdir(images_directory)
train_image_ids, val_image_ids = train_test_split(all_image_ids, test_size=0.15, random_state=42)
val_image_ids, test_image_ids = train_test_split(val_image_ids, test_size=0.1, random_state=42)

train_captions, val_captions, test_captions = [], [], []
for caption in captions_IDs:
    image_id, _ = caption.split('\t')
    if image_id in train_image_ids:
        train_captions.append(caption)
    elif image_id in val_image_ids:
        val_captions.append(caption)
    elif image_id in test_image_ids:
        test_captions.append(caption)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def extract_image_features(model, image_path):
    img = preprocess_image(image_path)
    features = model.predict(img, verbose=0)
    return features


base_model = InceptionV3(weights='imagenet', input_shape=(299, 299, 3))
base_model.layers.pop()
inception_v3_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
cnn_output_dim = inception_v3_model.output_shape[1]

train_image_features, val_image_features, test_image_features = {}, {}, {}

for img_name in tqdm(all_image_ids, desc="Processing Images", colour='green'):
    image_path = os.path.join(images_directory, img_name)
    try:
        image_features = extract_image_features(inception_v3_model, image_path)

        if img_name in train_image_ids:
            train_image_features[img_name] = image_features.flatten()
        elif img_name in val_image_ids:
            val_image_features[img_name] = image_features.flatten()
        elif img_name in test_image_ids:
            test_image_features[img_name] = image_features.flatten()
    except Exception as e:
        print(f"Error processing {img_name}: {e}")


def data_generator(captions, image_features, tokenizer, max_caption_length, batch_size):
    num_samples = len(captions)
    image_ids = list(image_features.keys())
    while True:
        np.random.shuffle(captions)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            X_images, X_captions, y = [], [], []

            for caption in captions[start_idx:end_idx]:
                image_id, caption_text = caption.split('\t')
                caption_text = caption_text.rstrip('\n')

                if image_id not in image_features:
                    continue

                seq = tokenizer.texts_to_sequences([caption_text])[0]

                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_caption_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    X_images.append(image_features[image_id])
                    X_captions.append(in_seq)
                    y.append(out_seq)

            if len(X_images) > 0:
                yield (np.array(X_images), np.array(X_captions)), np.array(y)


batch_size_train = 270
batch_size_val = 150


def build_model(vocab_size, max_caption_length, cnn_output_dim):
    input_image = Input(shape=(cnn_output_dim,), name='Features_Input')
    fe1 = BatchNormalization()(input_image)
    fe2 = Dense(256, activation='relu')(fe1)
    fe3 = BatchNormalization()(fe2)

    input_caption = Input(shape=(max_caption_length,), name='Sequence_Input')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(input_caption)
    se2 = LSTM(256)(se1)

    decoder1 = add([fe3, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax', name='Output_Layer')(decoder2)

    model = Model(inputs=[input_image, input_caption], outputs=outputs, name='Image_Captioning')
    return model


caption_model = build_model(vocab_size, max_caption_length, cnn_output_dim)
optimizer = Adam(learning_rate=0.01, clipnorm=1.0)
caption_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

output_signature = (
    (tf.TensorSpec(shape=(None, 2048), dtype=tf.float32),
     tf.TensorSpec(shape=(None, max_caption_length), dtype=tf.float32)),
    tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)
)

train_data_gen_instance = data_generator(train_captions, train_image_features, tokenizer, max_caption_length,
                                         batch_size_train)
val_data_gen_instance = data_generator(val_captions, val_image_features, tokenizer, max_caption_length, batch_size_val)

train_dataset = tf.data.Dataset.from_generator(
    lambda: train_data_gen_instance,
    output_signature=output_signature
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_data_gen_instance,
    output_signature=output_signature
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


def lr_scheduler(epoch, lr):
    return float(lr * tf.math.exp(-0.6))


lr_schedule = LearningRateScheduler(lr_scheduler)

print("Starting training...")
history = caption_model.fit(
    train_dataset,
    steps_per_epoch=len(train_captions) // batch_size_train,
    validation_data=val_dataset,
    validation_steps=len(val_captions) // batch_size_val,
    epochs=15,
    callbacks=[early_stopping, lr_schedule]
)

plt.figure(figsize=(15, 7), dpi=200)
sns.set_style('whitegrid')
plt.plot([x + 1 for x in range(len(history.history['loss']))], history.history['loss'], color='#E74C3C', marker='o')
plt.plot([x + 1 for x in range(len(history.history['loss']))], history.history['val_loss'], color='#641E16', marker='h')
plt.title('Train VS Validation', fontsize=15, fontweight='bold')
plt.legend(['Train Loss', 'Validation Loss'], loc='best')
plt.show()


def greedy_generator(image_features):
    in_text = 'start '
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length).reshape((1, max_caption_length))

        prediction = caption_model.predict([image_features.reshape(1, cnn_output_dim), sequence], verbose=0)
        idx = np.argmax(prediction)
        word = tokenizer.index_word[idx]

        in_text += ' ' + word
        if word == 'end':
            break

    in_text = in_text.replace('start ', '')
    in_text = in_text.replace(' end', '')
    return in_text


keys = list(test_image_features.keys())[:5]

for image_id in keys:
    feature = test_image_features[image_id]

    greedy_cap = greedy_generator(feature)

    print("-" * 30)
    print(f"Image: {image_id}")
    print("Greedy Caption:", greedy_cap)

    img_path = os.path.join(images_directory, image_id)
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()