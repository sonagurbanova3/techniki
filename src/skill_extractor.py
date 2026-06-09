from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import util

from preprocessing import clean_text
from model_loader import get_semantic_model


BLOCKED_WORDS = {
    "company", "name", "city", "state", "city state",
    "company city", "company city state",
    "work", "job", "required", "preferred", "candidate",
    "experience", "responsibilities", "include", "skills", "skill",
    "activities", "responsible", "procedures", "set",
    "site", "understand", "positioning",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "assigned", "encouraged", "served", "managed", "store",
    "parent", "parents", "student", "students", "start", "head",
    "com", "email", "gmail", "phone", "linkedin",
    "john", "jane", "smith", "matthew",
    "led", "deliver", "identify", "results",
    "senior", "manager", "projects", "functional",
    "cross", "cross functional", "enterprise",
}


def extract_candidate_phrases(text, max_phrases=25):
    """
    Automatycznie wyciąga ważne frazy z tekstu przy użyciu TF-IDF.
    Frazy są później traktowane jako kandydaci na kompetencje/słowa kluczowe.
    """
    clean = clean_text(text)

    if not clean:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_features=max_phrases,
    )

    try:
        vectorizer.fit_transform([clean])
    except ValueError:
        return []

    phrases = vectorizer.get_feature_names_out()
    filtered_phrases = []

    for phrase in phrases:
        phrase = phrase.strip().lower()

        if phrase in BLOCKED_WORDS:
            continue

        if any(char.isdigit() for char in phrase):
            continue

        if len(phrase) <= 2:
            continue

        filtered_phrases.append(phrase)

    return filtered_phrases


def compare_skills_semantically(cv_text, job_text, threshold=0.55, model=None):
    """
    Porównuje kompetencje z oferty z kompetencjami z CV.
    Używa embeddingów, więc znajduje także podobieństwa znaczeniowe,
    a nie tylko identyczne słowa.
    """
    cv_skills = extract_candidate_phrases(cv_text)
    job_skills = extract_candidate_phrases(job_text)

    found_skills = []
    missing_skills = []

    if len(cv_skills) == 0 or len(job_skills) == 0:
        return cv_skills, job_skills, found_skills, job_skills

    if model is None:
        model = get_semantic_model()

    cv_embeddings = model.encode(cv_skills, convert_to_tensor=True)
    job_embeddings = model.encode(job_skills, convert_to_tensor=True)

    for i, job_skill in enumerate(job_skills):
        similarities = util.cos_sim(job_embeddings[i], cv_embeddings)[0]

        best_score = similarities.max().item()
        best_index = similarities.argmax().item()

        if best_score >= threshold:
            found_skills.append({
                "required_skill": job_skill,
                "matched_cv_skill": cv_skills[best_index],
                "similarity": round(best_score, 2),
            })
        else:
            missing_skills.append(job_skill)

    return cv_skills, job_skills, found_skills, missing_skills
