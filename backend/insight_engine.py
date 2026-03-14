import pandas as pd

def generate_insights(df):

    insights = []

    high_income = df[df["income"] > 70000]["converted"].mean()
    low_income = df[df["income"] <= 70000]["converted"].mean()

    if high_income > low_income:
        ratio = round(high_income / low_income, 2)
        insights.append(f"Users with income > 70k convert {ratio}x more")

    high_visits = df[df["website_visits"] > 8]["converted"].mean()
    low_visits = df[df["website_visits"] <= 8]["converted"].mean()

    if high_visits > low_visits:
        ratio = round(high_visits / low_visits, 2)
        insights.append(f"Users visiting website > 8 times convert {ratio}x more")

    long_time = df[df["time_spent"] > 30]["converted"].mean()
    short_time = df[df["time_spent"] <= 30]["converted"].mean()

    if long_time > short_time:
        ratio = round(long_time / short_time, 2)
        insights.append(f"Users spending > 30 seconds convert {ratio}x more")

    return insights