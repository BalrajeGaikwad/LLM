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



🤖 What is a Transformer?
A Transformer is a deep learning model used to understand and generate human language (and even images, code, music now).
It doesn’t read word-by-word like older models — instead, it looks at the whole sentence at once and decides which words are most important using something called attention.

📦 Parts of a Transformer
It mainly has two blocks:

1. Encoder (understands the input):
Reads the input sentence like "I love mangoes".

Breaks it into words and finds the relationships between words using self-attention.

2. Decoder (generates output):
Takes the encoder’s understanding and starts generating the output sentence — like "मुझे आम पसंद हैं".

🧠 Key Superpower: Attention
Imagine reading this sentence:

“The mango I ate yesterday was sweet.”

To understand what was sweet, your brain knows it refers to mango — even though “mango” is far from “sweet”.

That’s what attention does: It lets the model focus on important words, no matter how far apart they are.

