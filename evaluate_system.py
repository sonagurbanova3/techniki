import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("src")

from evaluator import evaluate_ranking, expert_relevant_jobs, summarize_metrics
from model_loader import get_semantic_model
from preprocessing import clean_text
from recommender import score_single_job
from text_extractor import extract_text


PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATION_ROOT = PROJECT_ROOT / "data" / "raw" / "annotations" / "vacancy-resume-matching-dataset-main"
CV_DIR = ANNOTATION_ROOT / "CV"
VACANCIES_PATH = ANNOTATION_ROOT / "5_vacancies.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_expert_dataset():
    """Wczytuje 5 ofert pracy oraz 30 CV z anotacjami eksperckimi."""
    if not VACANCIES_PATH.exists():
        raise FileNotFoundError(f"Brak pliku: {VACANCIES_PATH}")

    vacancies_df = pd.read_csv(VACANCIES_PATH)
    vacancies_df = vacancies_df.reset_index(drop=True)
    vacancies_df["job_number"] = vacancies_df.index + 1

    required_columns = {"job_title", "job_description", "job_number"}

    if not required_columns.issubset(vacancies_df.columns):
        raise ValueError("5_vacancies.csv musi zawierać kolumny: job_title, job_description.")

    return vacancies_df


def baseline_tfidf_ranking(cv_text, vacancies_df):
    """Ranking baseline: samo TF-IDF + cosine similarity."""
    clean_cv = clean_text(cv_text)
    results = []

    for _, row in vacancies_df.iterrows():
        clean_job = clean_text(str(row["job_description"]))
        vectorizer = TfidfVectorizer(stop_words="english")

        try:
            tfidf_matrix = vectorizer.fit_transform([clean_cv, clean_job])
            score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except ValueError:
            score = 0.0

        results.append({
            "job_number": int(row["job_number"]),
            "job_title": row["job_title"],
            "score": round(score * 100, 2),
        })

    ranking_df = pd.DataFrame(results).sort_values("score", ascending=False)

    return ranking_df


def hybrid_ranking(cv_text, vacancies_df, model):
    """Ranking systemu: TF-IDF + Semantic + Skill Match."""
    clean_cv = clean_text(cv_text)
    cv_embedding = model.encode(clean_cv, convert_to_tensor=True)
    results = []

    for _, row in vacancies_df.iterrows():
        scores = score_single_job(
            cv_text=cv_text,
            job_description=str(row["job_description"]),
            model=model,
            cv_embedding=cv_embedding,
        )

        results.append({
            "job_number": int(row["job_number"]),
            "job_title": row["job_title"],
            "score": scores["Final Score"],
            "tfidf": scores["TF-IDF"],
            "semantic": scores["Semantic"],
            "skill_match": scores["Skill Match"],
        })

    ranking_df = pd.DataFrame(results).sort_values("score", ascending=False)

    return ranking_df


def evaluate_model_for_cv(cv_id, cv_text, vacancies_df, model=None):
    """Liczy metryki baseline i modelu hybrydowego dla jednego CV."""
    relevant_top1 = expert_relevant_jobs(cv_id=cv_id, top_n=1, mode="union")
    relevant_top3 = expert_relevant_jobs(cv_id=cv_id, top_n=3, mode="union")

    rows = []

    baseline_df = baseline_tfidf_ranking(cv_text, vacancies_df)
    baseline_prediction = baseline_df["job_number"].tolist()

    baseline_metrics_top1 = evaluate_ranking(
        baseline_prediction,
        relevant_top1,
        k_values=(1, 3, 5),
    )

    baseline_metrics_top3 = evaluate_ranking(
        baseline_prediction,
        relevant_top3,
        k_values=(1, 3, 5),
    )

    rows.append({
        "CV_ID": cv_id,
        "Model": "TF-IDF baseline",
        "Relevant definition": "expert top1 union",
        "Predicted ranking": baseline_prediction,
        **baseline_metrics_top1,
    })

    rows.append({
        "CV_ID": cv_id,
        "Model": "TF-IDF baseline",
        "Relevant definition": "expert top3 union",
        "Predicted ranking": baseline_prediction,
        **baseline_metrics_top3,
    })

    if model is None:
        model = get_semantic_model()

    hybrid_df = hybrid_ranking(cv_text, vacancies_df, model)
    hybrid_prediction = hybrid_df["job_number"].tolist()

    hybrid_metrics_top1 = evaluate_ranking(
        hybrid_prediction,
        relevant_top1,
        k_values=(1, 3, 5),
    )

    hybrid_metrics_top3 = evaluate_ranking(
        hybrid_prediction,
        relevant_top3,
        k_values=(1, 3, 5),
    )

    rows.append({
        "CV_ID": cv_id,
        "Model": "Hybrid model",
        "Relevant definition": "expert top1 union",
        "Predicted ranking": hybrid_prediction,
        **hybrid_metrics_top1,
    })

    rows.append({
        "CV_ID": cv_id,
        "Model": "Hybrid model",
        "Relevant definition": "expert top3 union",
        "Predicted ranking": hybrid_prediction,
        **hybrid_metrics_top3,
    })

    return rows


def create_entity_evaluation_template(sample_size=10):
    """
    Tworzy szablon do ręcznej oceny ekstrakcji encji.
    Tego nie da się uczciwie wypełnić automatycznie bez ręcznie oznaczonych encji.
    """
    rows = []

    for cv_id in range(1, sample_size + 1):
        rows.extend([
            {
                "CV_ID": cv_id,
                "Entity type": "skills",
                "Gold entities": "",
                "Predicted entities": "",
                "True positives": "",
                "False positives": "",
                "False negatives": "",
            },
            {
                "CV_ID": cv_id,
                "Entity type": "experience",
                "Gold entities": "",
                "Predicted entities": "",
                "True positives": "",
                "False positives": "",
                "False negatives": "",
            },
            {
                "CV_ID": cv_id,
                "Entity type": "education",
                "Gold entities": "",
                "Predicted entities": "",
                "True positives": "",
                "False positives": "",
                "False negatives": "",
            },
            {
                "CV_ID": cv_id,
                "Entity type": "certifications",
                "Gold entities": "",
                "Predicted entities": "",
                "True positives": "",
                "False positives": "",
                "False negatives": "",
            },
        ])

    return pd.DataFrame(rows)


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """
    Prosta konwersja DataFrame do tabeli Markdown bez dodatkowych bibliotek.
    Dzięki temu nie trzeba instalować pakietu tabulate.
    """
    if df.empty:
        return "_Brak danych._"

    columns = list(df.columns)

    header = "| " + " | ".join(str(column) for column in columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []

    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")

    return "\n".join([header, separator, *rows])


def write_markdown_report(results_df, summary_df, output_path):
    """Zapisuje krótki raport ewaluacji w Markdown."""
    lines = []

    lines.append("# Raport ewaluacji CV Matcher")
    lines.append("")
    lines.append("## Cel ewaluacji")
    lines.append(
        "Celem ewaluacji było porównanie systemu CV Matcher z baseline TF-IDF "
        "oraz sprawdzenie zgodności rankingu ofert z anotacjami eksperckimi."
    )
    lines.append("")

    lines.append("## Dane")
    lines.append(
        "Wykorzystano zbiór ekspercki zawierający 30 CV oraz 5 ofert pracy. "
        "Dla każdego CV dostępne są rankingi ofert przygotowane przez dwóch niezależnych anotatorów."
    )
    lines.append("")

    lines.append("## Modele")
    lines.append("- TF-IDF baseline: cosine similarity na reprezentacji TF-IDF.")
    lines.append(
        "- Hybrid model: 30% TF-IDF + 50% podobieństwo semantyczne Sentence Transformers "
        "+ 20% dopasowanie kompetencji."
    )
    lines.append("")

    lines.append("## Wyniki zagregowane")
    lines.append("")
    lines.append(dataframe_to_markdown_table(summary_df))
    lines.append("")

    lines.append("## Interpretacja")
    lines.append(
        "Wyższe wartości Precision@K, Recall@K, MAP@K, NDCG@K oraz MRR oznaczają "
        "lepsze dopasowanie rankingu systemu do oceny eksperckiej."
    )
    lines.append(
        "W raporcie należy porównać wyniki modelu hybrydowego z baseline TF-IDF "
        "i wskazać, czy dodanie embeddingów oraz dopasowania kompetencji poprawia jakość rekomendacji."
    )
    lines.append("")

    lines.append("## Ewaluacja ekstrakcji encji")
    lines.append(
        "Dla ekstrakcji encji przygotowano szablon "
        "`reports/manual_entity_evaluation_template.csv`. "
        "Należy ręcznie uzupełnić encje referencyjne i policzyć precision/recall "
        "dla typów: skills, experience, education, certifications."
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    vacancies_df = load_expert_dataset()
    model = get_semantic_model()

    all_rows = []

    for cv_id in range(1, 31):
        cv_path = CV_DIR / f"{cv_id}.docx"

        if not cv_path.exists():
            print(f"Pomijam CV {cv_id}: brak pliku {cv_path}")
            continue

        print(f"Ewaluacja CV {cv_id}/30...")
        cv_text = extract_text(str(cv_path))

        rows = evaluate_model_for_cv(
            cv_id=cv_id,
            cv_text=cv_text,
            vacancies_df=vacancies_df,
            model=model,
        )

        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)
    summary_df = summarize_metrics(results_df)

    results_path = REPORTS_DIR / "evaluation_results.csv"
    summary_path = REPORTS_DIR / "evaluation_summary.csv"
    report_path = REPORTS_DIR / "evaluation_report.md"
    entity_template_path = REPORTS_DIR / "manual_entity_evaluation_template.csv"

    results_df.to_csv(results_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    create_entity_evaluation_template(sample_size=10).to_csv(
        entity_template_path,
        index=False,
        encoding="utf-8",
    )

    write_markdown_report(results_df, summary_df, report_path)

    print("\nGotowe. Zapisano pliki:")
    print(f"- {results_path}")
    print(f"- {summary_path}")
    print(f"- {report_path}")
    print(f"- {entity_template_path}")

    print("\nPodsumowanie:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
