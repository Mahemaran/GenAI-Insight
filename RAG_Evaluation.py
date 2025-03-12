from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

# Define the context and generated text
context = 'the cat sat on the mat'
generated_text = 'the cat and dog sat on the floor'
# TF-IDF
# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
# Fit and transform the texts into TF-IDF matrices
tfidf_matrix = vectorizer.fit_transform([generated_text, context])
# Compute cosine similarity between the two texts
cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
# Output the cosine similarity
print(f"TF-IDF Score: {cosine_sim[0][0]}")
print('--------------')
# Compute BLEU score
bleu_score = sentence_bleu([generated_text], context)
print(f"BLEU Score: {bleu_score}")
print('--------------')

# Compute ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
scores = scorer.score(context, generated_text)
print(scores)
print('--------------')

# Initialize BERT-based model for semantic textual similarity
model = SentenceTransformer('all-MiniLM-L6-v2')
# Define the context and generated text
# context = "We are regarded as poor citizens, while the patricians are seen as good. What the authorities lavish upon themselves could relieve us..."
# generated_text = "We, the struggling citizens, stand in stark contrast to the prosperous patricians. Their abundance suffices them, but they overlook our plight..."

# Get the embeddings for both texts
context_embedding = model.encode([context])
generated_embedding = model.encode([generated_text])
# Compute cosine similarity based on embeddings
similarity = cosine_similarity(context_embedding, generated_embedding)
print(f"Semantic Cosine Similarity (Contextual): {similarity[0][0]}")