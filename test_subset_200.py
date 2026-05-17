import pandas as pd
import json
import ast
import requests
import time

def run_eval():
    csv_file = "subset_data_1200.csv"
    print(f"Loading {csv_file}...")
    
    try:
        # Read only the first 200 rows of the dataset
        df = pd.read_csv(csv_file).head(200)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {csv_file}. Make sure it is in the same folder.")
        return

    brier_scores = []
    print(f"Backtesting agent on {len(df)} real-world events from the Prophet dataset...\n")
    
    for index, row in df.iterrows():
        ticker = str(row['event_ticker'])
        title = str(row['title'])
        
        # Safely parse the string representations of lists and dicts from the CSV
        try:
            outcomes = ast.literal_eval(str(row['markets']))
            resolved_dict = json.loads(str(row['market_outcome']))
        except Exception as e:
            print(f"⚠️ Skipping {ticker} due to CSV parsing error: {e}")
            continue
            
        # Any outcome mapped to 1 in the dict is a winner
        actual_winners = [str(k).strip() for k, v in resolved_dict.items() if v == 1]
        
        # Prioritize the 'augmented_title' if it exists for richer context
        description_text = str(row['augmented_title']) if pd.notna(row['augmented_title']) else title
        
        # Construct the exact payload your agent.py expects
        payload = {
            "event_ticker": ticker,
            "market_ticker": ticker,
            "title": title,
            "description": description_text,
            "category": str(row['category']),
            "rules": str(row['rules']) if pd.notna(row['rules']) else "",
            "close_time": str(row['close_time']),
            "outcomes": outcomes
        }
        
        try:
            # Call your running agent.py server
            res = requests.post("http://localhost:8000/predict", json=payload)
            res.raise_for_status()
            prediction = res.json()
            
            probs = {str(p["market"]).strip(): p["probability"] for p in prediction.get("probabilities", [])}
            
            event_brier = 0.0
            for outcome in outcomes:
                outcome_clean = str(outcome).strip()
                # Fallback to uniform if agent missed an outcome
                p_i = probs.get(outcome_clean, 1.0 / len(outcomes)) 
                outcome_i = 1.0 if outcome_clean in actual_winners else 0.0
                event_brier += (p_i - outcome_i) ** 2
            
            brier_scores.append(event_brier)
            
            # Get the confidence assigned to the actual winner (or first winner if multiple)
            winner_prob = probs.get(actual_winners[0], 0.0) if actual_winners else 0.0
            
            print(f"🔹 {ticker} | Brier: {event_brier:.4f}")
            print(f"   Question: {title}")
            print(f"   Actual Winner(s): {actual_winners}")
            print(f"   Agent's Confidence in Winner: {winner_prob * 100:.2f}%\n")
            
        except Exception as e:
            print(f"❌ Failed to get prediction for {ticker}: {e}")

        # Sleep to respect Tavily and OpenRouter API rate limits
        time.sleep(1) 
        
    if brier_scores:
        avg_brier = sum(brier_scores) / len(brier_scores)
        print("="*60)
        print(f"🎯 PROPHET 200-EVENT SUBSET AVERAGE BRIER SCORE: {avg_brier:.4f}")
        print("="*60)

if __name__ == "__main__":
    run_eval()