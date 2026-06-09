def generate_expert_feedback(similarity_percent, found_skills, missing_skills):
    """
    Generuje opinię ekspercką na podstawie zasad HR:
    - dopasowanie doświadczenia,
    - kwalifikacje,
    - słowa kluczowe,
    - konkretność CV,
    - brakujące kompetencje.
    """

    found_count = len(found_skills)
    missing_count = len(missing_skills)

    if similarity_percent >= 50:
        level = "wysokie"
        summary = "CV jest dobrze dopasowane do wybranej oferty pracy."
    elif similarity_percent >= 25:
        level = "częściowe"
        summary = "CV jest częściowo dopasowane do wybranej oferty pracy."
    else:
        level = "niskie"
        summary = "CV jest słabo dopasowane do wybranej oferty pracy."

    feedback = f"""
    Ocena ekspercka CV:

    Poziom dopasowania: {level}
    Similarity score: {similarity_percent}%

    {summary}

    Analiza:
    System wykrył {found_count} kompetencji zgodnych z wymaganiami oferty oraz {missing_count} kompetencji brakujących.
    Ocena została przygotowana na podstawie kryteriów HR: zgodności doświadczenia, kwalifikacji, słów kluczowych oraz dopasowania profilu kandydata do wymagań stanowiska.

    Mocne strony:
    """

    if found_count > 0:
        feedback += "Kandydat posiada kompetencje istotne dla tej oferty, między innymi:\n"
        for skill in found_skills[:5]:
            if isinstance(skill, dict):
                feedback += f"- {skill['matched_cv_skill']} pasuje do wymagania: {skill['required_skill']}\n"
            else:
                feedback += f"- {skill}\n"
    else:
        feedback += "Nie wykryto istotnych kompetencji wspólnych z wymaganiami oferty.\n"

    feedback += "\nObszary do poprawy:\n"

    if missing_count > 0:
        feedback += "W CV warto uzupełnić lub mocniej podkreślić następujące elementy:\n"
        for skill in missing_skills[:7]:
            feedback += f"- {skill}\n"
    else:
        feedback += "Nie wykryto istotnych braków względem oferty.\n"

    feedback += """
    Rekomendacja:
    Kandydat powinien dopasować CV do konkretnej oferty, używając bardziej precyzyjnych słów kluczowych z ogłoszenia.
    Warto rozbudować opis doświadczenia zawodowego, projektów oraz osiągnięć, aby lepiej pokazać zgodność z wymaganiami stanowiska.
    """

    return feedback
