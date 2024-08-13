import os
import sys
import numpy as np
import json
import tensorflow as tf
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l1

# 在Dense层中添加L1正则化
dense_layer_with_l1 = Dense(128, activation='relu', kernel_regularizer=l1(0.01))
def factorial_tf(n):
    return tf.math.reduce_prod(tf.range(1, n + 1, dtype=tf.float32))

def polynomial_relu(x):
    # return 0.5 * x + 0.2 * tf.pow(x, 2) + 0.05 * tf.pow(x, 3)
    return x 

def polynomial_softmax(x, degree=3):
    poly_exp = sum(tf.math.pow(x, i) / factorial_tf(i) for i in range(degree + 1))
    poly_exp = tf.nn.relu(poly_exp)  
    return poly_exp / tf.reduce_sum(poly_exp, axis=-1, keepdims=True)

class PolynomialReLU(tf.keras.layers.Layer):
    def __init__(self):
        super(PolynomialReLU, self).__init__()

    def call(self, inputs):
        return polynomial_relu(inputs)

class PolynomialSoftmax(tf.keras.layers.Layer):
    def __init__(self, degree=3):
        super(PolynomialSoftmax, self).__init__()
        self.degree = degree

    def call(self, inputs):
        return polynomial_softmax(inputs, self.degree)

def fine_tune_model(model, train_data, train_labels, val_data, val_labels, epochs, batch_size, learning_rate):
    # for layer in model.layers[:-3]:
    #     layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_labels, 
              epochs=epochs, 
              batch_size=batch_size, 
              validation_data=(val_data, val_labels))

    return model

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, use_positional_encoding, max_seq_length, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_seq_length, embed_dim)
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.use_positional_encoding = use_positional_encoding

    def get_angles(self, pos, i, embed_dim):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        return pos * angle_rates

    def positional_encoding(self, position, embed_dim):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(embed_dim)[np.newaxis, :], embed_dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        if self.use_positional_encoding == 1:
            return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        elif self.use_positional_encoding == 0:
            print('***NOT USING POSITIONAL ENCODING***')
            return inputs

# Build the MH Self Attention.
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        print(f"Setting up {num_heads}-head self-attention layer with {embed_dim}-dimensional embeddings.")
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Make sure the embedding dimension is separable among heads.
        assert embed_dim % num_heads == 0

        # Setup the Q, K, V matrices.
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

        # 使用 PolynomialSoftmax
        self.polynomial_softmax = PolynomialSoftmax()

    # Compute attention score.
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # 使用多项式逼近Softmax
        weights = tf.nn.softmax(scaled_score,axis=-1)
        
        output = tf.matmul(weights, value)
        return output, weights

    # Separate the multi-head into individual heads.
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        print('==========================================')
        print(f'Projection dim: {self.projection_dim}; Query: {self.query_dense(inputs).shape}; Key: {self.key_dense(inputs).shape}; Value: {self.value_dense(inputs).shape}')

        # Instantiate the QKV layers.
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Separate the heads.
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        print(f'Single head dimensions:  Query: {query.shape}; Key: {key.shape}; Value: {value.shape}')

        # Calculate attention.
        attention, weights = self.attention(query, key, value)

        # Transpose the attention output.
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        # Concatenate the attention.
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)

        return output

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, attention_block_activation_type, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)

        # Instantiate FFN.
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu',kernel_regularizer=l1(0.1)),  # L1正则化
            Dense(embed_dim, kernel_regularizer=l1(0.1))  # L1正则化
        ])
                        
        self.layernorm1 = LayerNormalization(epsilon=1e-6, name="LayerNorm1")
        self.layernorm2 = LayerNormalization(epsilon=1e-6, name="LayerNorm2")
        self.dropout1 = Dropout(rate, name='LayerDropout1')
        self.dropout2 = Dropout(rate, name='LayerDropout2')

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_transformer_model(max_seq_length, use_positional_encoding, vocab_size, embed_dim, num_heads, attention_block_activation_type, ff_dim, num_layers, output_dim, rate=0.1):
    inputs = Input(shape=(max_seq_length,))
    embedding_layer = Embedding(vocab_size, embed_dim)(inputs)
    positional_encoding_layer = PositionalEncoding(use_positional_encoding, max_seq_length, embed_dim)(embedding_layer)
    transformer_blocks = []
    for _ in range(num_layers):
        transformer_blocks.append(TransformerBlock(embed_dim, num_heads, attention_block_activation_type, ff_dim, rate))
    transformer_output = positional_encoding_layer
    for transformer_block in transformer_blocks:
        transformer_output = transformer_block(transformer_output)
    pool = tf.keras.layers.GlobalAveragePooling1D(name='GlobalPool_Transformers')(transformer_output)
    dense = Dense(output_dim, activation=PolynomialSoftmax(), name='Dense_Classifier')(pool)  # 使用多项式逼近Softmax
    model = Model(inputs=inputs, outputs=dense)
    return model

#######################################################

def main():
    tf.config.set_visible_devices([], 'GPU')
    if len(sys.argv) < 2:
        print(f"USAGE: {sys.argv[0]} [option] [arguments]\n\
    -evaluate_DASHformer [Test sequences CSV] [Model .keras file] [Tokenizer file] [Max sequence length] [Predictions output file]\n")
        sys.exit(1)

    cmd_option = sys.argv[1]
    logging.basicConfig(filename='inference.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

    if cmd_option == "-fine_tune":
        if len(sys.argv) != 9:
            print(f'USAGE: {sys.argv[0]} {sys.argv[1]} [Training sequences CSV]  [Model .keras file to load] [Tokenizer file] [Max sequence length] [Epochs] [Batch size] [Learning rate]\n')
            sys.exit(1)

        protein_seq_file = sys.argv[2]
        keras_model_file = sys.argv[3]
        tokenizer_file = sys.argv[4]
        max_seq_length = int(sys.argv[5])
        epochs = int(sys.argv[6])
        batch_size = int(sys.argv[7])
        learning_rate = float(sys.argv[8])

        # 读取训练数据
        # ...（省略数据读取代码）
        if os.path.exists(protein_seq_file) == False:
            print(f'Could not find sequences CSV file {protein_seq_file}')

        if os.path.exists(keras_model_file) == False:
            print(f'Could not keras model file {keras_model_file}')

        # Assuming data_file format: sequence,class_label
        sequences = []
        class_labels = []

        with open(protein_seq_file, 'r') as file:
            for line in file:
                sequence, label = line.strip().split(',')
                sequences.append(sequence.split())  # Assuming sequences are whitespace-separated tokens
                class_labels.append(int(label))  # Assuming class labels are integers
                # if int(label) == 13 or int(label) == 3 or int(label) == 17 or int(label) == 11 or int(label) == 19 or int(label) == 21:
                #     sequences.append(sequence.split())  # Assuming sequences are whitespace-separated tokens
                #     class_labels.append(int(label))
        class_labels_onehot = to_categorical(class_labels)
        print(len(sequence))
        print(f"Shape of one-hot encoded class labels: {class_labels_onehot.shape}")

        # Define the classes while loading.
        print(f'Loading model from {keras_model_file}')
        model = load_model(keras_model_file, 
                           custom_objects={"PositionalEncoding": PositionalEncoding,
                                           "TransformerBlock": TransformerBlock,
                                           "MultiHeadSelfAttention": MultiHeadSelfAttention})
        
        with open('model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Tokenization and Padding starts below..
        print(f'Loading Tokenizer from {tokenizer_file}')
        with open(tokenizer_file, 'r') as file:
            tokenizer_config = file.read()

        # Load tokenizer from JSON configuration
        tokenizer = tokenizer_from_json(tokenizer_config)
        sequences_tokenized = tokenizer.texts_to_sequences(sequences)
        sequences_padded = pad_sequences(sequences_tokenized, maxlen=max_seq_length, padding='post')

        # Write the information on vocabulary, etc.
        vocab_size = len(tokenizer.word_index) + 1
        output_dim = len(set(class_labels))
        print(f"Vocab-size is {vocab_size}")
        # 加载预训练模型
        print(f'Loading model from {keras_model_file}')
        model = load_model(keras_model_file, custom_objects={"PositionalEncoding": PositionalEncoding,
                                                               "TransformerBlock": TransformerBlock,
                                                               "MultiHeadSelfAttention": MultiHeadSelfAttention})
        train_data, val_data, train_labels, val_labels = train_test_split(
            sequences_padded, class_labels_onehot, test_size=0.2, random_state=72,stratify=class_labels
        )

        _, train_class_counts = np.unique(np.argmax(train_labels,axis=1), return_counts=True)
        _, val_class_counts = np.unique(np.argmax(val_labels,axis=1), return_counts=True)
        print(np.argmax(val_labels,axis=1).shape)
        print("Training set class counts:", train_class_counts)
        print("Validation set class counts:", val_class_counts)

        # Fine-tuning 模型
        model = fine_tune_model(model, train_data, train_labels, 
                                val_data, val_labels, epochs, batch_size, learning_rate)

        # 保存 fine-tuned 模型
        model.save('fine_tuned.keras')



    # List the GPU devices, if any.
    # print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

    if cmd_option == "-evaluate_DASHformer":
        if len(sys.argv) != 7:
            print(f'USAGE: {sys.argv[0]} {sys.argv[1]} [Test sequences CSV] [Model .keras file] [Tokenizer file] [Max sequence length] [Predictions output file]\n')
            sys.exit(1)

        protein_seq_file = sys.argv[2]
        keras_model_file = sys.argv[3]
        tokenizer_file = sys.argv[4]
        max_seq_length = int(sys.argv[5])
        output_probs_file = sys.argv[6]

        if os.path.exists(protein_seq_file) == False:
            print(f'Could not find sequences CSV file {protein_seq_file}')

        if os.path.exists(keras_model_file) == False:
            print(f'Could not keras model file {keras_model_file}')

        # Assuming data_file format: sequence,class_label
        sequences = []
        class_labels = []

        with open(protein_seq_file, 'r') as file:
            for line in file:
                sequence, label = line.strip().split(',')
                sequences.append(sequence.split())  # Assuming sequences are whitespace-separated tokens
                class_labels.append(int(label))  # Assuming class labels are integers

        class_labels_onehot = to_categorical(class_labels)

        print(f"Shape of one-hot encoded class labels: {class_labels_onehot.shape}")

        # Define the classes while loading.
        print(f'Loading model from {keras_model_file}')
        model = load_model(keras_model_file, 
                           custom_objects={"PositionalEncoding": PositionalEncoding,
                                           "TransformerBlock": TransformerBlock,
                                           "MultiHeadSelfAttention": MultiHeadSelfAttention})
        
        with open('model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Tokenization and Padding starts below..
        print(f'Loading Tokenizer from {tokenizer_file}')
        with open(tokenizer_file, 'r') as file:
            tokenizer_config = file.read()

        # Load tokenizer from JSON configuration
        tokenizer = tokenizer_from_json(tokenizer_config)
        sequences_tokenized = tokenizer.texts_to_sequences(sequences)
        
        sequences_padded = pad_sequences(sequences_tokenized, maxlen=max_seq_length, padding='post')

        # Write the information on vocabulary, etc.
        vocab_size = len(tokenizer.word_index) + 1
        output_dim = len(set(class_labels))
        print(f"Vocab-size is {vocab_size}")

        prediction_results = model.predict(sequences_padded)
        np.save("pre.npy",prediction_results)
        eval_results = model.evaluate(sequences_padded, class_labels_onehot)
         
        print("Loss:", eval_results[0])
        logging.info(eval_results[0])
        print("Accuracy:", eval_results[1])
        logging.info(eval_results[1])

        with open(output_probs_file, 'w') as file:
            file.write("Sequence,Label,Prediction\n")
            for sequence, label, prediction in zip(sequences, class_labels, prediction_results):
                file.write(f"{sequence},{label},{np.argmax(prediction)}\n")
                if label != np.argmax(prediction):
                    print(label,np.argmax(prediction))

        ###############################################################################################

    
if __name__ == '__main__':
    main()
