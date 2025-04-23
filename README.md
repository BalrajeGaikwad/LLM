# LLM


from tensorflow.keras.preprocessing.text import Tokenizer

texts = ["I love NLP", "NLP is awesome"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Encode
sequences = tokenizer.texts_to_sequences(texts)
print("Encoded:", sequences)

# Decode
word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}
decoded_texts = [[index_word[idx] for idx in seq] for seq in sequences]
print("Decoded:", decoded_texts)
