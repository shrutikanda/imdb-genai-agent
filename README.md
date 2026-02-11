# 🎬 IMDB Movie Agent

A Gen AI powered conversational voice agent that uses the IMDB Top 1000 dataset to answer various questions about movies.

## Features

- **Hybrid Search**: Combines BM25 keyword search + Vector semantic search with Reciprocal Rank Fusion (RRF)
- **Voice Input**: Record questions using your microphone
- **Smart Filtering**: Filter by genre, year, rating, meta score, director, actors, gross earnings
- **Actor Clarification**: Asks if you want lead roles (Star1) or any role (Star1-4)
- **Recommendations**: Suggests similar movies based on rating and meta score

## Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | OpenAI GPT-4o (via OpenAI Agents SDK) |
| **Backend** | Node.js + Express |
| **Frontend** | Streamlit |
| **Vector Store** | Custom implementation with pre-computed embeddings |
| **Search** | Hybrid BM25 + Vector search with RRF |
| **Embeddings** | OpenAI text-embedding-3-small |

## Prerequisites

- Node.js 18+
- Python 3.10+
- OpenAI API Key

## Installation

### 1. Extract the project

```bash
cd agentfeedback
```

### 2. Install Node.js dependencies

```bash
cd inventory-node
npm install
```

### 3. Install Python dependencies

```bash
cd Python
pip install -r requirements.txt
```

### 4. Set up environment variables

Create `Python/.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Create `inventory-node/.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Application

### Step 1: Start the Node.js API Server

```bash
cd inventory-node
node server.js
```

Server starts at `http://localhost:3000`

### Step 2: Start the Streamlit UI

```bash
# In a new terminal
cd Python
streamlit run movieagent.py
```

UI opens at `http://localhost:8501`

## Test Questions

| # | Question | Feature Tested |
|---|----------|----------------|
| 1 | When did The Matrix release? | Basic search |
| 2 | Top 5 movies of 2019 by meta score | Year filter + sorting |
| 3 | Top 7 comedy movies between 2010-2020 by IMDB rating | Genre + year range |
| 4 | Top horror movies with meta > 85 and rating > 8 | Multiple filters |
| 5 | Top directors with $500M+ movies at least twice | Aggregation query |
| 6 | Top 10 movies with over 1M votes but lower gross | Underrated movies |
| 7 | Comedy movies about death | Semantic search + genre |
| 8 | Summarize Spielberg's sci-fi movies | Director + genre + synthesis |
| 9 | Movies before 1990 about police | Semantic search (not keyword) |
| 10 | Al Pacino movies with rating > 8 | Actor clarification flow |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /movies/search` | Hybrid search with filters |
| `GET /movies/top-directors` | Directors with high-grossing movies |
| `GET /movies/underrated` | High votes, low gross movies |
| `GET /movies/recommendations` | Similar movies by rating/meta |

### Example API Calls

```bash
# Search for police movies before 1990
curl "http://localhost:3000/movies/search?q=police%20investigation&year_end=1989"

# Get top directors with $500M+ movies
curl "http://localhost:3000/movies/top-directors?min_gross=500000000&min_count=2"

# Get underrated movies
curl "http://localhost:3000/movies/underrated?min_votes=1000000&limit=10"

# Get recommendations
curl "http://localhost:3000/movies/recommendations?avg_rating=8.5&avg_meta_score=85"
```

## Project Structure

```
agentfeedback/
├── inventory-node/
│   ├── server.js          # Express API server
│   ├── vectorStore.js     # Hybrid search implementation
│   ├── indexMovies.js     # One-time embedding generator
│   ├── embeddings.json    # Pre-computed vector embeddings
│   ├── imdb_top_1000.csv  # Movie dataset
│   └── package.json       # Node dependencies
├── Python/
│   ├── agent.py           # Function tools + Agent definition
│   ├── app.py             # Streamlit UI
│   ├── movieagent.py      # Entry point (imports agent + app)
│   ├── requirements.txt   # Python dependencies
│   └── .env               # API keys
└── README.md
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│  OpenAI Agent   │────▶│  Node.js API    │
│   (Voice/Text)  │     │  (GPT-4o)       │     │  (Hybrid Search)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                        ┌─────────────────────────────────────────┐
                        │           Hybrid Search Engine           │
                        ├─────────────────┬───────────────────────┤
                        │   BM25 Search   │   Vector Search       │
                        │   (MiniSearch)  │   (text-embedding-3)  │
                        └────────┬────────┴───────────┬───────────┘
                                 └──────────┬─────────┘
                                            ▼
                                 ┌─────────────────┐
                                 │  RRF Fusion     │
                                 │  (k=60)         │
                                 └─────────────────┘
```

## Model Used

- **LLM**: OpenAI GPT-4o via OpenAI Agents SDK
- **Embeddings**: text-embedding-3-small (1536 dimensions, pre-computed)

## Nice-to-Have Features

✅ **Actor Clarification**: Bot asks "lead actor or any role?" before searching  
✅ **Recommendations**: Suggests similar quality movies after answering queries

## License

MIT
