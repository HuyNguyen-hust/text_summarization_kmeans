from sentence_transformers import SentenceTransformer
import joblib

model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
joblib.dump(model, 'paraphrase-xlm-r-multilingual-v1.pkl')
