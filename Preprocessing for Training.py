import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import pickle
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
    return text

# Usage
file_path = "Test_Resumes.pdf"
extracted_text = pdf_reader(file_path)
split_text = extracted_text.split("")

# Create a DataFrame for the split resumes
df = pd.DataFrame({'resume_text': split_text})

# Clean text: remove special characters, punctuation, and numbers
df['cleaned_text'] = df['resume_text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# Convert text to lowercase
df['cleaned_text'] = df['cleaned_text'].str.lower()

# Tokenize sentences into words
df['tokenized_text'] = df['cleaned_text'].apply(word_tokenize)

# Remove stopwords (common words like 'the', 'and', 'in')
stop_words = set(stopwords.words('english'))
df['filtered_text'] = df['tokenized_text'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Apply stemming
df['stemmed_text'] = df['filtered_text'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

# Apply lemmatization
df['lemmatized_text'] = df['filtered_text'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])

# Pickling the data
with open('data.pkl', 'wb') as file:
    pickle.dump(df["lemmatized_text"], file)
