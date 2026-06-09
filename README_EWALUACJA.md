# Ewaluacja projektu CV Matcher

Ten plik opisuje sposób uruchomienia i interpretacji ewaluacji systemu CV Matcher.

## Co jest mierzone?

System porównuje dwa podejścia:

1. **TF-IDF baseline** — klasyczne podobieństwo tekstów na podstawie reprezentacji TF-IDF.
2. **Hybrid model** — wynik używany w aplikacji:
   - 30% TF-IDF,
   - 50% podobieństwo semantyczne Sentence Transformers,
   - 20% dopasowanie kompetencji.

## Dane ewaluacyjne

Skrypt używa danych z folderu:

```text
data/raw/annotations/vacancy-resume-matching-dataset-main/