from rag_evaluator import RAGEvaluator
from agent import run_agent
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
question = "When did The Matrix release?"
reference = "The Matrix was released in 1999."


conversation = []
response, conversation = run_agent(conversation, question)

print("\nAGENT RESPONSE:\n", response)

evaluator = RAGEvaluator()
metrics = evaluator.evaluate_all(question, response, reference)

print("\n--- METRICS ---")
for k, v in metrics.items():
    print(f"{k}: {v}")