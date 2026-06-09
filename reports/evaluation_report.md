# Raport ewaluacji CV Matcher

## Cel ewaluacji
Celem ewaluacji było porównanie systemu CV Matcher z baseline TF-IDF oraz sprawdzenie zgodności rankingu ofert z anotacjami eksperckimi.

## Dane
Wykorzystano zbiór ekspercki zawierający 30 CV oraz 5 ofert pracy. Dla każdego CV dostępne są rankingi ofert przygotowane przez dwóch niezależnych anotatorów.

## Modele
- TF-IDF baseline: cosine similarity na reprezentacji TF-IDF.
- Hybrid model: 30% TF-IDF + 50% podobieństwo semantyczne Sentence Transformers + 20% dopasowanie kompetencji.

## Wyniki zagregowane

| Model | Precision@1 | Recall@1 | MAP@1 | NDCG@1 | Precision@3 | Recall@3 | MAP@3 | NDCG@3 | Precision@5 | Recall@5 | MAP@5 | NDCG@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hybrid model | 0.3167 | 0.1136 | 0.3167 | 0.3167 | 0.5111 | 0.5519 | 0.4195 | 0.5298 | 0.5767 | 1.0 | 0.6246 | 0.7352 | 0.5978 |
| TF-IDF baseline | 0.4167 | 0.1581 | 0.4167 | 0.4167 | 0.5333 | 0.5756 | 0.4718 | 0.5782 | 0.5767 | 1.0 | 0.6633 | 0.768 | 0.6639 |

## Interpretacja
Wyższe wartości Precision@K, Recall@K, MAP@K, NDCG@K oraz MRR oznaczają lepsze dopasowanie rankingu systemu do oceny eksperckiej.
W raporcie należy porównać wyniki modelu hybrydowego z baseline TF-IDF i wskazać, czy dodanie embeddingów oraz dopasowania kompetencji poprawia jakość rekomendacji.

## Ewaluacja ekstrakcji encji
Dla ekstrakcji encji przygotowano szablon `reports/manual_entity_evaluation_template.csv`. Należy ręcznie uzupełnić encje referencyjne i policzyć precision/recall dla typów: skills, experience, education, certifications.
