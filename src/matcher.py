
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import clean_text

def rank_job_offers(cv_text, jobs_df, top_n=10):

    clean_cv_text = clean_text(cv_text)

    results = []

    vectorizer = TfidfVectorizer(stop_words='english')

    for index, row in jobs_df.iterrows():

        job_title = row["Job Title"]

        job_description = row["Description"]

        clean_job_text = clean_text(job_description)

        documents = [clean_cv_text, clean_job_text]

        tfidf_matrix = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )[0][0]

        results.append({
            "Job Index": index,
            "Job Title": job_title,
            "Similarity": round(similarity * 100, 2)
        })

    ranking_df = pd.DataFrame(results)

    ranking_df = ranking_df.sort_values(
        by="Similarity",
        ascending=False
    )

    return ranking_df.head(top_n)
