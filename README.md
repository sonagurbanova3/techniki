# CV Matcher - dopasowanie CV do ofert pracy

## Przedmiot

Techniki i narzędzia przetwarzania języka naturalnego

## Temat projektu

**CV Matcher - dopasowanie CV do ofert pracy**

## Zespół D

- Sofiia Hurbanova
- Emilia Walczyk
- Katarzyna Jakimiuk

---

## 1. Opis projektu

Proces dopasowywania kandydatów do ofert pracy jest często czasochłonny, ręczny i subiektywny. Kandydaci nie zawsze potrafią ocenić, czy ich profil odpowiada wymaganiom stanowiska, a rekruterzy analizują dużą liczbę dokumentów, co utrudnia szybkie porównanie aplikacji.

Celem projektu jest stworzenie systemu wykorzystującego metody NLP, który automatycznie analizuje treść CV oraz ofert pracy, ocenia poziom dopasowania i generuje rekomendacje dla użytkownika.

System wspiera kandydatów w ocenie dopasowania CV do konkretnej oferty oraz pomaga wskazać kompetencje, które warto uzupełnić lub mocniej podkreślić w dokumencie aplikacyjnym.

---

## 2. Główne funkcjonalności

Aplikacja realizuje następujące funkcje:

- ekstrakcja tekstu z CV w formatach PDF, DOCX, TXT oraz obrazów,
- automatyczne wykrywanie sekcji CV, takich jak doświadczenie, edukacja, umiejętności i certyfikaty,
- analiza treści ofert pracy,
- obliczanie dopasowania CV do wybranej oferty,
- generowanie rankingu najlepiej dopasowanych ofert,
- wykrywanie kompetencji obecnych w CV i wymaganych w ofercie,
- wskazywanie brakujących kompetencji,
- generowanie krótkiej analizy dopasowania oraz sugestii poprawy CV,
- ewaluacja systemu i porównanie z baseline TF-IDF.

---

## 3. Wykorzystane metody NLP

W projekcie zastosowano następujące techniki przetwarzania języka naturalnego:

### 3.1. Czyszczenie tekstu

Tekst CV i ofert pracy jest normalizowany poprzez:

- zmianę liter na małe,
- usuwanie zbędnych znaków specjalnych,
- ograniczenie nadmiarowych spacji,
- przygotowanie tekstu do dalszej analizy.

### 3.2. TF-IDF

TF-IDF jest wykorzystywany jako klasyczna metoda reprezentacji tekstu. Pozwala ocenić podobieństwo CV i oferty pracy na podstawie wspólnych ważnych słów oraz fraz.

TF-IDF pełni również rolę baseline w ewaluacji systemu.

### 3.3. Sentence Transformers

Do oceny podobieństwa semantycznego wykorzystano model:

```text
all-MiniLM-L6-v2