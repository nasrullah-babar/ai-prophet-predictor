# Prophet Hacks Superforecaster

This is a custom prediction market agent built for the Prophet Arena hackathon. The goal of this project isn't just to make an LLM guess outcomes, but to explicitly minimize Brier Score penalties using a hybrid architecture of AI extraction and deterministic mathematical guardrails.

During local testing against the 1,200-event Prophet Arena benchmark subset, this agent successfully dodged common LLM hallucination traps (like misreading Supreme Court vote tallies or hallucinating future events) and achieved highly competitive sub-0.10 baseline scores on resolved events.

## The Architecture (The "Secret Sauce")

Large Language Models are great at reading text, but they are terrible at probability math under pressure. This agent splits the labor:

1. **Two-Pass Search (Tavily API):**
   * **Pass 1 (The Fact Finder):** Searches for exact, conclusive results for past/resolved events.
   * **Pass 2 (The Forecast Finder):** If the event is in the future, it runs a secondary query specifically targeting live betting odds, polling, and expert consensus.
2. **The LLM Forensic Engine (Brain):**
   * The LLM does zero math. It is strictly instructed to act as a forensic text investigator.
   * It executes a **Temporal Date Check** to ensure it isn't reading old news.
   * It executes a **Vote Breakdown Check** to separate majority vs. dissenting counts (preventing the classic trap where an LLM sees a "6-3 decision" and blindly guesses 6).
3. **Deterministic Brier Clamps (Brawn):**
   * If the LLM confirms a winner with hard proof, Python assigns `0.95`.
   * If the event is in the future, Python extracts the live odds but mathematically **clamps the confidence at `0.75`**.
   * If there is a total blind spot, Python falls back to the LLM's historical base-rate knowledge but **clamps confidence at `0.60`**. 
   * *Why?* Because in Brier scoring, being confidently wrong destroys your average. These clamps mathematically prevent catastrophic 1.8+ error penalties.

## Getting Started

### Prerequisites
You will need API keys for OpenRouter and Tavily Search.

```bash
OPENROUTER_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here

Installation
Clone the repository.

Install the required dependencies:

Bash
pip install -r requirements.txt
Run the FastAPI server:

Bash
uvicorn agent:app --host 0.0.0.0 --port 8000
Endpoints
GET /health
A simple wake-up endpoint to ensure the server is alive and ready to receive traffic (useful for cold-starts on free hosting tiers).

POST /predict
The main forecasting engine. Accepts the standard Prophet Arena event JSON shape.

Example Payload:

JSON
{
  "event_ticker": "task-001",
  "market_ticker": "task-001",
  "title": "Who will win: Pittsburgh or Atlanta?",
  "description": "Predict the winner of the scheduled matchup.",
  "category": "Sports",
  "rules": "Resolves to the official winner after the game is final.",
  "close_time": "2026-03-21T23:59:59Z",
  "outcomes": ["Pittsburgh", "Atlanta"]
}
Example Response:

JSON
{
  "probabilities": [
    { "market": "Pittsburgh", "probability": 0.60 },
    { "market": "Atlanta",    "probability": 0.40 }
  ],
  "p_yes": 0.60,
  "p_no": 0.40
}
Tech Stack
FastAPI / Uvicorn: For high-performance async serving.

OpenRouter (gpt-4o-mini): For fast, cost-effective forensic text extraction.

Tavily Search API: For AI-native web scraping that bypasses SEO spam.