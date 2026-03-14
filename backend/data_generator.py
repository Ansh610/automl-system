import pandas as pd
import numpy as np

def generate_dataset(n=3000):

    np.random.seed(42)

    age = np.random.randint(18, 65, n)
    income = np.random.randint(20000, 120000, n)

    city = np.random.choice(
        ["Delhi", "Mumbai", "Bangalore"],
        n,
        p=[0.4, 0.35, 0.25]
    )

    gender = np.random.choice(["Male", "Female"], n)

    website_visits = np.random.randint(1, 20, n)
    time_spent = np.random.randint(5, 120, n)

    # ---------- Normalize ----------
    income_score = income / 120000
    visit_score = website_visits / 20
    time_score = time_spent / 120
    age_score = age / 65

    # ---------- City effect ----------
    city_effect = np.zeros(n)

    city_effect[city == "Bangalore"] = 0.08
    city_effect[city == "Mumbai"] = 0.04
    city_effect[city == "Delhi"] = -0.02

    # ---------- Gender effect ----------
    gender_effect = np.zeros(n)
    gender_effect[gender == "Female"] = 0.03

    # ---------- Interaction effects ----------
    interaction = (
        0.10 * visit_score * time_score +
        0.05 * income_score * visit_score
    )

    # ---------- Final probability ----------
    probability = (
        0.25 * income_score +
        0.30 * visit_score +
        0.25 * time_score +
        0.10 * age_score +
        city_effect +
        gender_effect +
        interaction
    )

    # ---------- Add noise ----------
    noise = np.random.normal(0, 0.08, n)

    probability = probability + noise

    probability = np.clip(probability, 0, 1)

    # ---------- Generate target ----------
    converted = np.random.binomial(1, probability)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "city": city,
        "gender": gender,
        "website_visits": website_visits,
        "time_spent": time_spent,
        "converted": converted
    })

    return df