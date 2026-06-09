import sys
import tempfile

import pandas as pd
import streamlit as st

sys.path.append("src")

from cv_parser import parse_cv_sections
from llm_feedback import generate_expert_feedback
from recommender import recommend_jobs, score_single_job
from text_extractor import extract_text


st.set_page_config(
    page_title="CV Matcher",
    page_icon="📄",
    layout="wide",
)



jobs_path = "data/raw/jobs/jobs_matching_cv_categories.csv"

try:
    jobs_df = pd.read_csv(jobs_path)
except FileNotFoundError:
    st.error(
        "Nie znaleziono pliku z ofertami pracy. Sprawdź, czy uruchamiasz aplikację z folderu CV_Matcher_Project."
    )
    st.stop()

required_columns = {"Job Title", "Description"}
if not required_columns.issubset(jobs_df.columns):
    st.error("Plik z ofertami pracy musi zawierać kolumny: Job Title oraz Description.")
    st.stop()



st.title("📄 CV Matcher - dopasowanie CV do ofert pracy")
st.write("System analizuje CV użytkownika i sprawdza dopasowanie do ofert pracy.")

st.metric(
    label="Liczba dostępnych ofert pracy",
    value=len(jobs_df),
)



st.subheader("Dostępne oferty pracy")

job_titles = jobs_df["Job Title"].tolist()
selected_job_title = st.selectbox("Wybierz ofertę pracy", job_titles)

selected_job = jobs_df[jobs_df["Job Title"] == selected_job_title].iloc[0]



st.subheader("Szczegóły oferty")
st.write("###", selected_job["Job Title"])
st.write(str(selected_job["Description"])[:4000])



st.subheader("Wgraj CV")

uploaded_file = st.file_uploader(
    "Wgraj CV",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
)



if uploaded_file is not None:
    file_extension = "." + uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.read())
        cv_path = tmp_file.name

    try:
        cv_text = extract_text(cv_path)
    except Exception as exc:
        st.error(f"Nie udało się odczytać CV: {exc}")
        st.stop()

    if not cv_text.strip():
        st.warning("Nie udało się wyciągnąć tekstu z CV. Spróbuj użyć innego pliku.")
        st.stop()

    st.subheader("Tekst wyciągnięty z CV")
    st.text_area("Podgląd CV", cv_text[:3000], height=250)

    cv_sections = parse_cv_sections(cv_text)

    st.subheader("Wykryte sekcje CV")

    for section_name, section_text in cv_sections.items():
        if section_text.strip():
            with st.expander(section_name):
                st.write(section_text[:1500])

    if st.button("Analizuj dopasowanie"):
        with st.spinner("Analizuję CV i ofertę pracy. Pierwsze uruchomienie może potrwać dłużej..."):
            selected_scores = score_single_job(
                cv_text=cv_text,
                job_description=str(selected_job["Description"]),
            )

        st.subheader("Wynik dopasowania do wybranej oferty")

        final_score = selected_scores["Final Score"]
        st.metric(label="Final Score", value=f"{final_score}%")
        st.progress(min(max(int(final_score), 0), 100))

        col1, col2, col3 = st.columns(3)
        col1.metric("TF-IDF", f"{selected_scores['TF-IDF']}%")
        col2.metric("Semantic", f"{selected_scores['Semantic']}%")
        col3.metric("Skill Match", f"{selected_scores['Skill Match']}%")

        st.caption(
            "Final Score = 30% TF-IDF + 50% Semantic + 20% Skill Match."
        )

        cv_skills = selected_scores["CV Skills"]
        job_skills = selected_scores["Job Skills"]
        found_skills = selected_scores["Found Skills"]
        missing_skills = selected_scores["Missing Skills"]

        st.subheader("Kompetencje wykryte automatycznie w CV")
        st.write(cv_skills)

        st.subheader("Kompetencje wykryte automatycznie w ofercie")
        st.write(job_skills)

        st.subheader("Dopasowane kompetencje")
        if len(found_skills) > 0:
            st.dataframe(pd.DataFrame(found_skills), use_container_width=True)
        else:
            st.warning("Nie znaleziono dopasowanych kompetencji.")

        st.subheader("Brakujące kompetencje")
        if len(missing_skills) > 0:
            st.error(", ".join(missing_skills))
        else:
            st.success("Nie wykryto brakujących kompetencji.")

        st.subheader("Opinia ekspercka")
        expert_feedback = generate_expert_feedback(
            final_score,
            found_skills,
            missing_skills,
        )
        st.info(expert_feedback)

        st.subheader("Inne potencjalnie dopasowane oferty")

        with st.spinner("Tworzę ranking ofert..."):
            recommended_jobs = recommend_jobs(
                cv_text,
                jobs_df,
                top_n=5,
            )

        score_columns = ["Job Title", "TF-IDF", "Semantic", "Skill Match", "Final Score"]
        st.dataframe(recommended_jobs[score_columns], use_container_width=True)

        st.subheader("Szczegóły rekomendacji")

        for _, row in recommended_jobs.iterrows():
            with st.expander(f"{row['Job Title']} — Final Score: {row['Final Score']}%"):
                st.write("### Skąd wziął się wynik?")
                st.write(f"**TF-IDF:** {row['TF-IDF']}%")
                st.write("Ten wynik pokazuje podobieństwo słów między CV a ofertą.")

                st.write(f"**Semantic:** {row['Semantic']}%")
                st.write(
                    "Ten wynik pokazuje podobieństwo znaczenia tekstów na podstawie embeddingów Sentence Transformers."
                )

                st.write(f"**Skill Match:** {row['Skill Match']}%")
                st.write(
                    "Ten wynik pokazuje, jaka część kompetencji z oferty została odnaleziona lub semantycznie dopasowana w CV."
                )

                st.write(f"**Final Score:** {row['Final Score']}%")
                st.write("Final Score jest wynikiem hybrydowym: 30% TF-IDF + 50% Semantic + 20% Skill Match.")

                st.write("### Kompetencje wykryte w CV")
                st.write(row["CV Skills"])

                st.write("### Kompetencje wymagane w ofercie")
                st.write(row["Job Skills"])

                st.write("### Dopasowane kompetencje")
                st.write(row["Found Skills"])

                st.write("### Brakujące kompetencje")
                st.write(row["Missing Skills"])

                st.write("### Fragment opisu oferty")
                st.write(row["Description"][:1500])
