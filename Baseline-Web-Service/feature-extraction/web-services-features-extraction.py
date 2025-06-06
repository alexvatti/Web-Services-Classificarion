import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


def vectorize_descriptions(service_list, max_features=1000):
    """
    Converts a list of service descriptions into a DataFrame of TF-IDF features.
    Limits to top `max_features` terms by importance across the corpus.
    """
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(service_list)
    feature_names = vectorizer.get_feature_names_out()
    
    return pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)



def embed_descriptions_with_sbert(services, model_name='all-MiniLM-L6-v2'):
    """
    Converts service descriptions into SBERT embeddings using SentenceTransformer.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(services, show_progress_bar=True)
    
    return pd.DataFrame(embeddings, index=services.index if isinstance(services, pd.Series) else None)

if __name__ == "__main__":
    for n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        input_csv = f"../data-preprocessing/Pre_Processed_Top_{n}_Web_Services_Categories.csv"
        df = pd.read_csv(input_csv)

        # Ensure the column exists and fill missing with empty strings
        df['Pre-Processed Description'] = df['Pre-Processed Description'].fillna("")

        tfidf_df = vectorize_descriptions(df['Pre-Processed Description'])
        tfidf_output_csv = f"Pre_Processed_Top_{n}_Web_Services_Categories_TFIDF.csv"
        tfidf_df.to_csv(tfidf_output_csv, encoding='utf-8', index=False, header=True)
        print(f"TF-IDF features saved to: {tfidf_output_csv}")

        embedding_df = embed_descriptions_with_sbert(df['Pre-Processed Description'])
        embedding_output_csv = f"Pre_Processed_Top_{n}_Web_Services_Categories_SBERT_Embeddings.csv"
        embedding_df.to_csv(embedding_output_csv, encoding='utf-8', index=False, header=True)
        print(f"SBERT embeddings saved to: {embedding_output_csv}")