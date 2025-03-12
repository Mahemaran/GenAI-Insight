import gensim.downloader as api
# Load a prebuilt Word2Vec model
model = api.load("word2vec-google-news-300")
# given word for converting embedding
word = "Guru"
embedding = model[word]
print(embedding)