"""
Analysing climate sentiment in TV weather forecasts with an LLM
================================================================

Companion script for the "Using AI Programmatically" section of the
Harnessing AI course. It drives the Gemini API over a set of TV weather
forecast captions (one bulletin per day, every June from 2020 to 2025),
asking the model to summarise each one, judge its mood, and extract any
practical recommendation, then compares the mood and recommendations
year on year.

Designed to run in Google Colab, but works locally too.

Setup
-----
1. Get a free Gemini API key from https://aistudio.google.com/apikey
2. In Colab: add it via the Secrets panel (key icon) as GEMINI_API_KEY,
   with notebook access ON.
   Locally: set it as an environment variable, e.g.
       export GEMINI_API_KEY="AIza...your key..."   (macOS / Linux)
       $env:GEMINI_API_KEY = "AIza...your key..."   (Windows PowerShell)
3. Place weather_world_captions.csv (columns: date, identifier, caption) next to
   this script, or upload it to your Colab session.

Install dependencies (run once):
    pip install -U google-genai pydantic pandas seaborn
"""

import os
from enum import Enum

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Step 1 - Load the API key
# ---------------------------------------------------------------------------
# Read the key from Colab Secrets; fall back to a prompt if not on Colab.
try:
    from google.colab import userdata
    os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
    print("Loaded GEMINI_API_KEY from Colab Secrets.")
except Exception:
    if not os.environ.get("GEMINI_API_KEY"):
        import getpass
        os.environ["GEMINI_API_KEY"] = getpass.getpass("Paste your Gemini API key: ")
        print("Loaded GEMINI_API_KEY from prompt.")
    else:
        print("Loaded GEMINI_API_KEY from environment.")

# The client automatically picks up the GEMINI_API_KEY environment variable.
client = genai.Client()

# A quick smoke test to confirm the key and connection work.
resp = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Reply with exactly the word: connected",
)
print(resp.text)


# ---------------------------------------------------------------------------
# Step 2 - The data
# ---------------------------------------------------------------------------
# 180 forecasts: the evening bulletin for every day in June, 2020-2025.
# Columns: date, identifier, caption
forecasts = pd.read_csv("weather_world_captions.csv", parse_dates=["date"])
print(forecasts.head())


# ---------------------------------------------------------------------------
# Step 3 - Tell the model what to do, and what shape the answer should be
# ---------------------------------------------------------------------------
system_instruction = (
    "You analyse the captions of TV weather forecasts. "
    "For each caption: write a very short summary, judge the overall emotional mood "
    "conveyed to viewers, and extract any practical safety recommendation given to the public."
)


class Mood(str, Enum):
    anxious = "anxious"        # alarmed, worried, warning tone
    neutral = "neutral"        # matter-of-fact, calm
    reassuring = "reassuring"  # pleasant, comforting tone


class Recommendation(str, Enum):
    stay_indoors = "stay_indoors"
    keep_cool = "keep_cool"
    keep_warm = "keep_warm"
    none = "none"


class ForecastAnalysis(BaseModel):
    summary: str = Field(description="A 4-6 word summary of the forecast")
    mood: Mood = Field(description="Overall mood conveyed to viewers")
    recommendation: Recommendation = Field(
        description="The main practical advice to the public, or 'none' if there is no safety advice"
    )


# ---------------------------------------------------------------------------
# Step 4 - Run the model over every caption
# ---------------------------------------------------------------------------
# One call per item, carrying date and identifier through so each result stays
# linked to its source row.
results = []
for row in forecasts.itertuples(index=False):
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=row.caption,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ForecastAnalysis,
        ),
    )
    analysis = response.parsed  # a ForecastAnalysis instance
    results.append(
        {
            "date": row.date,
            "identifier": row.identifier,
            "caption": row.caption,
            "summary": analysis.summary,
            "mood": analysis.mood.value,
            "recommendation": analysis.recommendation.value,
        }
    )

output_data = pd.DataFrame(results)
print(output_data.head())


# ---------------------------------------------------------------------------
# Step 5 - Compare the years (every forecast is from June, so differences
#          across years reflect change over time, not the seasons)
# ---------------------------------------------------------------------------
output_data["year"] = output_data["date"].dt.year

# Mood by year - count of each mood label in each year's 30 forecasts.
mood_by_year = (
    output_data
    .groupby(["year", "mood"])
    .size()
    .unstack(fill_value=0)
)
print(mood_by_year)

# Recommendations by year - the same breakdown for the practical advice given.
rec_by_year = (
    output_data
    .groupby(["year", "recommendation"])
    .size()
    .unstack(fill_value=0)
)
print(rec_by_year)

# Plot the two side by side. Seaborn expects "long" (tidy) data, so we melt
# the per-year count tables first; each panel shows grouped bars per year.
sns.set_theme(style="whitegrid")

mood_long = mood_by_year.reset_index().melt(
    id_vars="year", var_name="mood", value_name="count"
)
rec_long = rec_by_year.reset_index().melt(
    id_vars="year", var_name="recommendation", value_name="count"
)

fig, (ax_mood, ax_rec) = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=mood_long, x="year", y="count", hue="mood", ax=ax_mood)
ax_mood.set_title("Mood of forecasts by year (June only)")
ax_mood.set_xlabel("Year")
ax_mood.set_ylabel("Number of forecasts")

sns.barplot(data=rec_long, x="year", y="count", hue="recommendation", ax=ax_rec)
ax_rec.set_title("Recommendations by year (June only)")
ax_rec.set_xlabel("Year")
ax_rec.set_ylabel("Number of forecasts")

plt.tight_layout()
plt.show()

# The headline question: is "stay indoors" recommended more often over time?
stay_indoors_by_year = (
    output_data[output_data["recommendation"] == "stay_indoors"]
    .groupby("year")
    .size()
)
print(stay_indoors_by_year)
