from preprocessing import clean_text


SECTION_HEADERS = {
    "summary": [
        "summary", "profile", "objective", "professional summary",
        "career summary"
    ],
    "experience": [
        "experience", "work experience", "employment history",
        "work history", "professional experience"
    ],
    "education": [
        "education", "academic background", "qualifications",
        "degrees"
    ],
    "skills": [
        "skills", "highlights", "technical skills",
        "core competencies", "competencies"
    ],
    "certifications": [
        "certifications", "certificates", "licenses",
        "licences", "training"
    ]
}


def detect_section(line):
    clean_line = clean_text(line)

    for section_name, headers in SECTION_HEADERS.items():
        for header in headers:
            if clean_line == header or clean_line.startswith(header):
                return section_name

    return None


def parse_cv_sections(cv_text):
    """
    Funkcja próbuje podzielić CV na główne sekcje:
    summary, experience, education, skills, certifications.

    Dzięki temu system nie zakłada jednego konkretnego układu CV.
    """

    sections = {
        "summary": "",
        "experience": "",
        "education": "",
        "skills": "",
        "certifications": "",
        "other": ""
    }

    current_section = "other"

    lines = cv_text.splitlines()

    for line in lines:
        stripped_line = line.strip()

        if not stripped_line:
            continue

        detected = detect_section(stripped_line)

        if detected:
            current_section = detected
            continue

        sections[current_section] += stripped_line + "\n"

    return sections
