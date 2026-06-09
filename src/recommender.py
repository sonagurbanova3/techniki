import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util

from preprocessing import clean_text
from skill_extractor import compare_skills_semantically
from model_loader import get_semantic_model


TFIDF_WEIGHT = 0.30
SEMANTIC_WEIGHT = 0.50
SKILL_WEIGHT = 0.20


def calculate_tfidf_score(clean_cv, clean_job):
    """Liczy podobieństwo TF-IDF dla CV i opisu oferty."""
    if not clean_cv or not clean_job:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        tfidf_matrix = vectorizer.fit_transform([clean_cv, clean_job])
    except ValueError:
        return 0.0

    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def score_single_job(cv_text, job_description, model=None, cv_embedding=None):
    """
    Liczy pełny wynik dopasowania CV do jednej oferty.
    Zwraca składowe: TF-IDF, semantic, skill match oraz final score.
    """
    if model is None:
        model = get_semantic_model()

    clean_cv = clean_text(cv_text)
    clean_job = clean_text(job_description)

    tfidf_score = calculate_tfidf_score(clean_cv, clean_job)

    if cv_embedding is None:
        cv_embedding = model.encode(clean_cv, convert_to_tensor=True)

    job_embedding = model.encode(clean_job, convert_to_tensor=True)
    semantic_score = util.cos_sim(cv_embedding, job_embedding).item()

    cv_skills, job_skills, found_skills, missing_skills = compare_skills_semantically(
        cv_text,
        job_description,
        model=model,
    )

    if len(job_skills) > 0:
        skill_score = len(found_skills) / len(job_skills)
    else:
        skill_score = 0.0

    final_score = (
        TFIDF_WEIGHT * tfidf_score
        + SEMANTIC_WEIGHT * semantic_score
        + SKILL_WEIGHT * skill_score
    )

    return {
        "TF-IDF": round(tfidf_score * 100, 2),
        "Semantic": round(semantic_score * 100, 2),
        "Skill Match": round(skill_score * 100, 2),
        "Final Score": round(final_score * 100, 2),
        "CV Skills": cv_skills,
        "Job Skills": job_skills,
        "Found Skills": found_skills,
        "Missing Skills": missing_skills,
    }


def recommend_jobs(cv_text, jobs_df, top_n=5):
    """Tworzy ranking najlepiej dopasowanych ofert pracy dla podanego CV."""
    clean_cv = clean_text(cv_text)
    model = get_semantic_model()
    cv_embedding = model.encode(clean_cv, convert_to_tensor=True)

    results = []

    for index, row in jobs_df.iterrows():
        job_title = row["Job Title"]
        job_description = str(row["Description"])

        scores = score_single_job(
            cv_text=cv_text,
            job_description=job_description,
            model=model,
            cv_embedding=cv_embedding,
        )

        results.append({
            "Job Index": index,
            "Job Title": job_title,
            **scores,
            "Description": job_description,
        })

    ranking_df = pd.DataFrame(results)

    ranking_df = ranking_df.sort_values(
        by="Final Score",
        ascending=False,
    )

    return ranking_df.head(top_n)
