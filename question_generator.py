# question_generator.py
# Generates interview questions using OpenAI Chat API with .env support and fallback templates.

import os
import json
import random
from typing import List
from dotenv import load_dotenv

# Load variables from .env file (if exists)
load_dotenv()

try:
    import openai
except Exception:
    openai = None

DEFAULT_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4o-mini")

FALLBACK_TEMPLATES = {
    "Software": [
        "Explain the software development lifecycle and your role in a recent project.",
        "What is SOLID? Give an example of the Single Responsibility Principle.",
        "Describe how you'd debug a production performance issue.",
        "How do you approach writing unit tests?",
        "Explain REST vs GraphQL and when you'd choose each.",
        "Describe a challenging technical decision you made.",
        "How do you manage technical debt in a team?",
        "Explain dependency injection and why it's useful.",
        "Walk me through optimizing a slow SQL query.",
        "Where do you see distributed systems failures coming from?"
    ],
    "Data Science": [
        "Describe a machine learning project you worked on.",
        "How do you handle missing data during preprocessing?",
        "Explain bias-variance tradeoff with an example.",
        "Walk me through feature engineering for tabular data.",
        "How do you evaluate models with imbalanced classes?",
        "Explain cross-validation and why it's used.",
        "How do you tune hyperparameters for XGBoost?",
        "Describe how you'd deploy a model.",
        "Explain overfitting to a non-technical stakeholder.",
        "How do you ensure reproducibility of experiments?"
    ],
    "Product": [
        "How would you design a product feature from idea to launch?",
        "How do you prioritize features when resources are limited?",
        "Describe how you'd gather and use customer feedback.",
        "Explain A/B testing and how you'd interpret results.",
        "How do you write measurable success metrics for a feature?",
        "Tell me about a time you changed direction due to research.",
        "How do you work with engineering to scope a risky feature?",
        "Describe how you'd approach pricing for a new product.",
        "Explain how you'd handle conflicting stakeholder requests.",
        "How do you ensure product accessibility?"
    ],
    "Other": []
}

DEFAULT_GENERIC = [
    "Tell me about yourself and why you're a good fit for this role.",
    "Describe a time you faced a challenge and how you solved it.",
    "Explain a technical concept you recently learned.",
    "How do you prioritize tasks with multiple deadlines?",
    "Describe a project where you had to collaborate closely.",
    "Tell me about a failure and what you learned.",
    "How do you handle feedback from peers?",
    "What motivates you in your work?",
    "How do you keep your technical skills up to date?",
    "Describe a time you had to learn a new tool quickly."
]


def _fallback_questions(domain: str, n: int) -> List[str]:
    domain = domain or "Other"
    domain_key = domain if domain in FALLBACK_TEMPLATES else "Other"
    pool = FALLBACK_TEMPLATES.get(domain_key) or DEFAULT_GENERIC
    if len(pool) >= n:
        return pool[:n]
    pool_extended = pool + DEFAULT_GENERIC
    random.shuffle(pool_extended)
    return pool_extended[:n]


def generate_questions(domain: str = "Software",
                       level: str = "Intermediate",
                       n: int = 10,
                       temperature: float = 0.7) -> List[str]:
    """
    Generate interview questions for given domain and level.
    Uses OpenAI API if available; falls back to templates otherwise.
    """
    domain = (domain or "Other").title()
    level = (level or "Intermediate").title()

    api_key = os.getenv("OPENAI_API_KEY")
    if openai is None or api_key is None:
        return _fallback_questions(domain, n)

    openai.api_key = api_key
    model = DEFAULT_MODEL

    system_prompt = (
        "You are an expert interviewer. Produce a JSON array of concise interview questions "
        "appropriate for the given domain and expertise level. Each question max 25 words."
    )
    user_prompt = (
        f"Domain: {domain}\nLevel: {level}\nNumber of questions: {n}\n"
        "Output only a JSON array of strings, e.g. [\"Q1\", \"Q2\", ...]"
    )

    try:
        res = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=800,
        )
        text = res['choices'][0]['message']['content'].strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and parsed:
                return [str(x).strip() for x in parsed][:n]
        except Exception:
            lines = [ln.strip(" -\t") for ln in text.splitlines() if ln.strip()]
            if lines:
                return lines[:n]
        return _fallback_questions(domain, n)
    except Exception:
        return _fallback_questions(domain, n)
