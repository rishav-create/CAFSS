import easyocr
import nltk
from rapidfuzz import fuzz
from nltk.corpus import stopwords, opinion_lexicon
import numpy as np
import os

nltk.download('stopwords', quiet=True)
nltk.download('opinion_lexicon', quiet=True)

class ConfidenceAwareFuzzySentimentScoring:

    def __init__(self, alpha=0.55, beta=0.45, theta_p=0.05, theta_n=-0.05, min_sim=0.70):
        self.alpha = alpha
        self.beta = beta
        self.theta_p = theta_p
        self.theta_n = theta_n
        self.min_sim = min_sim

        self.stop_words = set(stopwords.words('english'))
        self.positive_words = list(opinion_lexicon.positive())
        self.negative_words = list(opinion_lexicon.negative())

        print("Loading OCR model...")
        self.reader = easyocr.Reader(['en'], gpu=False)

    # ------------------ PREPROCESSING ------------------

    def preprocess(self, texts, confidences):

        filtered_texts = []
        filtered_conf = []

        for text, conf in zip(texts, confidences):

            words = text.strip().split()

            for word in words:

                w = word.lower().strip()

                if w.isalpha() and w not in self.stop_words and len(w) > 2:
                    filtered_texts.append(w)
                    filtered_conf.append(conf)

        return filtered_texts, filtered_conf

    # ------------------ FUZZY SENTIMENT ------------------

    def fuzzy_sentiment_similarity(self, word):

        # Direct lexicon check (very strong signal)
        if word in self.positive_words:
            return 1.0, 0.0

        if word in self.negative_words:
            return 0.0, 1.0

        # Fuzzy similarity with lexicon
        fp_scores = [fuzz.ratio(word, p)/100.0 for p in self.positive_words]
        fn_scores = [fuzz.ratio(word, n)/100.0 for n in self.negative_words]

        Fp = max(fp_scores)
        Fn = max(fn_scores)

        if Fp < self.min_sim:
            Fp = 0.0

        if Fn < self.min_sim:
            Fn = 0.0

        return Fp, Fn

    # ------------------ MAIN ANALYSIS ------------------

    def analyze(self, image_path):

        if not os.path.exists(image_path):
            return "Error", 0.0, "Image not found!"

        # Step 1: OCR
        result = self.reader.readtext(image_path)

        T = [item[1] for item in result]
        C = [item[2] for item in result]

        print(f"\n🔍 OCR extracted {len(T)} regions: {T}")

        # Step 2: Preprocessing
        T_prime, C_prime = self.preprocess(T, C)

        print(f"✅ After filtering: {T_prime}")

        if not T_prime:
            return "Neutral", 0.0, "No valid tokens"

        # Step 3: Sentiment scoring
        print("\n📊 PER-WORD DEBUG")
        print("Word\t\tCi\tFp\tFn\tSi")
        print("-" * 50)

        scores = []

        for wi, Ci in zip(T_prime, C_prime):

            Fp, Fn = self.fuzzy_sentiment_similarity(wi)

            Si = Ci * (self.alpha * Fp - self.beta * Fn)

            scores.append(Si)

            print(f"{wi:<12}\t{Ci:.3f}\t{Fp:.3f}\t{Fn:.3f}\t{Si:.4f}")

        # Step 4: Aggregation
        S = np.mean(scores)

        print(f"\n📈 Final Aggregated Score S = {S:.4f}")

        # Step 5: Classification
        if S >= self.theta_p:
            L = "Positive"

        elif S <= self.theta_n:
            L = "Negative"

        else:
            L = "Neutral"

        return L, round(float(S), 4), f"Tokens: {len(T_prime)}"


# ================= RUN HERE =================

if __name__ == "__main__":

    cafss = ConfidenceAwareFuzzySentimentScoring(
        alpha=0.55,
        beta=0.45,
        theta_p=0.05,
        theta_n=-0.05,
        min_sim=0.70
    )

    image_path = "amazon_dataset2.jpg"

    label, score, info = cafss.analyze(image_path)   

    print("=" * 60)
    print(f"🎯 FINAL SENTIMENT LABEL : {label}")
    print(f"📊 Aggregated Score       : {score}")
    print("=" * 60)