import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    return " ".join(lemmatized_tokens)


if __name__ == "__main__":
    for n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        input_csv = f"../data/top_web_services_categories_output/Top_{n}_Web_Services_Categories.csv"
        df = pd.read_csv(input_csv)

        # Apply preprocessing
        print(df)
        df['Pre-Processed Description'] = df['Service Description'].apply(preprocess_text)

        # Save the result
        output_csv = f"Pre_Processed_Top_{n}_Web_Services_Categories.csv"
        df.to_csv(output_csv, encoding='utf-8', index=False, header=True)
        print(df)

        print(f"Preprocessing done! Processed file saved to: {output_csv}")
