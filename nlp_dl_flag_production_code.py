
import pandas as pd
import numpy as np
import re
import pickle as pkl
import nltk
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


class DIFlagModel:

    def __init__(self, file_path=None):
        self.file_path = file_path
        self.vectorizer = None
        self.model = None
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
    # -----------------------------
    # Data Loading
    # -----------------------------

    def load_data(self):
        try:
            df = pd.read_csv(self.file_path, encoding='latin1')

            data = df[['ObsFullDescription', 'DI_flag']]

            data.dropna(subset=['ObsFullDescription'], inplace=True)

            data['DI_flag_Labels'] = data['DI_flag'].apply(
                lambda x: 1 if x == 'Yes' else 0
            )

            data = data.drop_duplicates(
                subset=['ObsFullDescription', 'DI_flag_Labels']
            )

            data.reset_index(drop=True, inplace=True)

            return data

        except Exception as e:
            print("Error while loading data")
            print(e)

    # -----------------------------
    # Text Preprocessing Functions
    # -----------------------------
    def html_remove(self, text):
        return re.sub('<.*?>', ' ', text)

    def special_char_remove(self, text):
        return re.sub('[^a-zA-Z]', ' ', text)

    def lower_case(self, text):
        text = text.lower().strip()
        text = re.sub(' +', ' ', text)
        return text

    def distinct_words(self, text):
        words = text.split()

        unique_words = []

        for word in words:
            if word not in unique_words:
                unique_words.append(word)

        return " ".join(unique_words)

    def remove_stopwords(self, text):

        stop_words = set(stopwords.words('english'))

        negative_words = ['not']

        stop_words = stop_words - set(negative_words)

        words = text.split()

        words = [word for word in words if word not in stop_words]

        return " ".join(words)

    def lemmatization(self, text):

        wnl = WordNetLemmatizer()

        words = text.split()

        words = [wnl.lemmatize(word, pos='v') for word in words]

        return " ".join(words)

    # -----------------------------
    # Complete Preprocessing
    # -----------------------------
    def preprocess_text(self, text):

        text = self.html_remove(text)
        text = self.special_char_remove(text)
        text = self.lower_case(text)
        text = self.distinct_words(text)
        text = self.remove_stopwords(text)
        text = self.lemmatization(text)

        return text

    # -----------------------------
    # Data Preparation
    # -----------------------------
    def prepare_data(self, data):

        try:

            data['processed_text'] = data['ObsFullDescription'].apply(
                self.preprocess_text
            )

            X = data['processed_text']
            y = data['DI_flag_Labels']

            return train_test_split(
                X,
                y,
                test_size=0.30,
                random_state=42
            )

        except Exception as e:
            print("Error during preprocessing")
            print(e)

    # -----------------------------
    # Model Training
    # -----------------------------
    def train_model(self, X_train, y_train):

        try:

            self.vectorizer = CountVectorizer()

            X_train_bow = self.vectorizer.fit_transform(X_train)

            self.model = LogisticRegression(class_weight='balanced',max_iter=1000,random_state=42)

            self.model.fit(X_train_bow, y_train)

            #print("Model Training Completed")

        except Exception as e:
            print("Error while training model")
            print(e)

    # -----------------------------
    # Model Evaluation
    # -----------------------------
    def evaluate_model(self, X_train, X_test, y_train, y_test):

        try:

            X_train_bow = self.vectorizer.transform(X_train)
            X_test_bow = self.vectorizer.transform(X_test)

            train_pred = self.model.predict(X_train_bow)
            test_pred = self.model.predict(X_test_bow)

            print("\nTraining Accuracy :",
                  accuracy_score(y_train, train_pred))

            print("\nTesting Accuracy :",
                  accuracy_score(y_test, test_pred))

            print("\nClassification Report")
            print(classification_report(y_test, test_pred))

        except Exception as e:
            print("Error during evaluation")
            print(e)

    # -----------------------------
    # Save Artifacts
    # -----------------------------
    def save_model(self):

        try:

            pkl.dump(
                self.model,
                open("nlp_DI_flag_model.sav", "wb")
            )

            pkl.dump(
                self.vectorizer,
                open("nlp_DI_flag_vectorizer.sav", "wb")
            )

            print("Model Saved Successfully")

        except Exception as e:
            print("Error while saving model")
            print(e)

    def load_artifacts(
        self,
        model_path="nlp_DI_flag_model.sav",
        vectorizer_path="nlp_DI_flag_vectorizer.sav"
    ):
        try:
            model_path = Path(model_path)
            vectorizer_path = Path(vectorizer_path)

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}"
                )
            if not vectorizer_path.exists():
                raise FileNotFoundError(
                    f"Vectorizer file not found: {vectorizer_path}"
                )

            self.model = pkl.load(model_path.open("rb"))
            self.vectorizer = pkl.load(vectorizer_path.open("rb"))

            return self

        except Exception as e:
            print("Error while loading artifacts")
            print(e)
            raise

    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        if self.model is None or self.vectorizer is None:
            raise ValueError(
                "Model artifacts are not loaded. Call load_artifacts() first."
            )

        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        raw_prediction = self.model.predict(X)[0]
        predicted_label = "Yes" if raw_prediction == 1 else "No"

        confidence = None
        if hasattr(self.model, "predict_proba"):
            confidence = self.model.predict_proba(X)[0].max()

        return {
            "dl_flag_label": predicted_label,
            "raw_prediction": int(raw_prediction),
            "confidence": float(confidence) if confidence is not None else None,
            "processed_text": processed_text,
        }

    # -----------------------------
    # Complete Pipeline
    # -----------------------------
    def run(self):

        try:

            data = self.load_data()

            X_train, X_test, y_train, y_test = self.prepare_data(data)

            self.train_model(X_train, y_train)

            self.evaluate_model(
                X_train,
                X_test,
                y_train,
                y_test
            )

            self.save_model()

        except Exception as e:
            print("Pipeline Execution Failed")
            print(e)


# =====================================
# Main Execution
# =====================================

if __name__ == "__main__":

    obj = DIFlagModel(
        r"C:\Users\chandini ch\Downloads\AQWA_BASE_Data.csv"
    )

    obj.run()
