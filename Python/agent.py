"""
IMDB Movie Agent - Function Tools and Agent Definition
"""
from pathlib import Path
from dotenv import load_dotenv
from agents import Agent, Runner, TResponseInputItem, function_tool, ModelSettings
from pydantic import BaseModel
import requests

# Load .env from the same folder as this file
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# API Base URL
API_BASE = "http://localhost:3000"


# ============ FUNCTION TOOLS ============

@function_tool
def search_movies(
    q: str,
    genre: str = None,
    year: int = None,
    year_start: int = None,
    year_end: int = None,
    min_rating: float = None,
    min_meta_score: int = None,
    director: str = None,
    star: str = None,
    star1: str = None,
    min_votes: int = None,
    min_gross: int = None,
    sort_by: str = None,
    order: str = "desc",
    limit: int = 10
) -> dict:
    """
    Search movies using hybrid search (BM25 + Vector + RRF).
    
    Args:
        q: Search query (required) - e.g., "police investigation", "comedy death"
        genre: Filter by genre - e.g., "comedy", "horror", "sci-fi"
        year: Exact year - e.g., 2019
        year_start: Year range start - e.g., 2010
        year_end: Year range end - e.g., 2020
        min_rating: Minimum IMDB rating - e.g., 8.0
        min_meta_score: Minimum meta score - e.g., 85
        director: Filter by director name - e.g., "Steven Spielberg"
        star: Filter by any actor (Star1-4) - e.g., "Al Pacino"
        star1: Filter by lead actor only (Star1) - e.g., "Al Pacino"
        min_votes: Minimum number of votes - e.g., 1000000
        min_gross: Minimum gross earnings - e.g., 50000000
        sort_by: Sort field - "IMDB_Rating", "Meta_score", "Gross", "Released_Year"
        order: Sort order - "desc" or "asc"
        limit: Max results to return - default 10
    
    Returns:
        dict with search results including title, year, genre, rating, overview, etc.
    """
    params = {"q": q, "limit": limit, "order": order}
    
    if genre:
        params["genre"] = genre
    if year:
        params["year"] = year
    if year_start:
        params["year_start"] = year_start
    if year_end:
        params["year_end"] = year_end
    if min_rating:
        params["min_rating"] = min_rating
    if min_meta_score:
        params["min_meta_score"] = min_meta_score
    if director:
        params["director"] = director
    if star:
        params["star"] = star
    if star1:
        params["star1"] = star1
    if min_votes:
        params["min_votes"] = min_votes
    if min_gross:
        params["min_gross"] = min_gross
    if sort_by:
        params["sort_by"] = sort_by
    
    try:
        response = requests.get(f"{API_BASE}/movies/search", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "message": "Failed to connect to movie API"}


@function_tool
def get_top_directors(
    min_gross: int = 500000000,
    min_count: int = 2,
    limit: int = 10
) -> dict:
    """
    Get directors with multiple high-grossing movies.
    
    Args:
        min_gross: Minimum gross per movie in dollars (default $500M = 500000000)
        min_count: Minimum number of movies meeting the gross threshold (default 2)
        limit: Max directors to return (default 10)
    
    Returns:
        Directors with their high-grossing movies, sorted by movie count
    """
    params = {"min_gross": min_gross, "min_count": min_count, "limit": limit}
    
    try:
        response = requests.get(f"{API_BASE}/movies/top-directors", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "message": "Failed to connect to movie API"}


@function_tool
def get_underrated_movies(
    min_votes: int = 1000000,
    limit: int = 10
) -> dict:
    """
    Get movies with high votes but lower gross earnings (underrated gems).
    
    Args:
        min_votes: Minimum number of votes (default 1M = 1000000)
        limit: Max movies to return (default 10)
    
    Returns:
        Movies sorted by lowest gross (potential underrated gems)
    """
    params = {"min_votes": min_votes, "limit": limit}
    
    try:
        response = requests.get(f"{API_BASE}/movies/underrated", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "message": "Failed to connect to movie API"}


@function_tool
def get_recommendations(
    avg_rating: float = 8.0,
    avg_meta_score: float = 80,
    exclude_titles: str = None,
    limit: int = 5
) -> dict:
    """
    Get movie recommendations based on similar IMDB rating and Meta score.
    
    Args:
        avg_rating: Target IMDB rating to find similar movies (e.g., 8.5)
        avg_meta_score: Target Meta score to find similar movies (e.g., 85)
        exclude_titles: Comma-separated movie titles to exclude from recommendations
        limit: Max recommendations to return (default 5)
    
    Returns:
        Movies with similar ratings/meta scores
    """
    params = {"limit": limit}
    
    if avg_rating:
        params["avg_rating"] = avg_rating
    if avg_meta_score:
        params["avg_meta_score"] = avg_meta_score
    if exclude_titles:
        params["exclude"] = exclude_titles
    
    try:
        response = requests.get(f"{API_BASE}/movies/recommendations", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "message": "Failed to get recommendations"}


# ============ AGENT OUTPUT ============

class AgentOutput(BaseModel):
    Message: str

# ============ AGENT INSTRUCTIONS ============

AGENT_INSTRUCTIONS = """You are an IMDB movie expert assistant with the Top 1000 IMDB movies database.
                        
                        You are an IMDB movie database assistant. You ONLY answer questions using the tools provided.
                        and you MUST follow the important restrictions below. Always use the tools to find information before answering,
                        and NEVER make up information that is not in the database. If information is not there simply say - I don't have information
                        outside provided tools.

                        ═══════════════════════════════════════════════════════════════════════════════
                        CRITICAL RULE - MANDATORY TOOL USAGE:
                        ═══════════════════════════════════════════════════════════════════════════════
                            
                        You MUST call a tool BEFORE providing ANY movie information. 
                        NEVER answer from your own knowledge - your training data is NOT the database.
                            
                        For EVERY movie-related question:
                            1. FIRST call the appropriate tool (usually search_movies)
                            2. WAIT for the tool result
                            3. ONLY use information from the tool result in your response
                            4. If the tool returns no results or an error, say "No results found in the database"
                            
                        NOTE: DO NOT supplement tool results with your own knowledge.
                        NOTE: DO NOT mention movies that were not returned by the tool.

                            1. ONLY answer questions about movies in the database (Top 1000 IMDB movies)
                            2. DO NOT answer questions about:
                            - Movies not in the tools results
                            - Hindi/Bollywood movies (not in database)
                            - TV shows, web series
                            - General knowledge questions
                            - Anything unrelated to the movie database

                            3. If user asks about something outside the database.tools results, respond:
                            "I can only answer questions about movies in my database (Top 1000 IMDB movies). 
                            This database contains primarily English-language Hollywood films. 
                            I don't have information about Hindi/Bollywood movies or other content outside this dataset."

                            4. ALWAYS use the provided tools to search before answering
                            5. NEVER make up movie information - only use what the tools return

                        TOOLS AVAILABLE:
                        1. search_movies - Find/filter movies using hybrid search (keyword + semantic)
                        2. get_top_directors - Get directors with multiple high-grossing movies
                        3. get_underrated_movies - Get high-vote movies with low gross (underrated gems)
                        4. get_recommendations - Get similar movies based on rating/meta score

                        RESPONSE FORMATTING RULES:
                        - Your response MUST only include movies that appear in the tool's "results" array
                        - Use EXACT titles, years, ratings from the tool output - do not modify or add to them
                        - If the tool returns 4 movies, mention only those 4 movies - never add more
                        - Count the results in the tool output and report that exact count to the user

                        CRITICAL - ACTOR QUERIES:
                        When user asks about movies featuring an actor (e.g., "Al Pacino movies", "Tom Hanks films"), 
                        you MUST ask this clarification question BEFORE making any tool call:

                        "Are you looking for movies where [Actor Name] is the lead actor (Star1) or movies where [Actor Name] appears in any role (Star1-4)?"

                        Wait for user response, then:
                        - If user says "lead" or "main actor" -> use q="Actor Name" AND star1="Actor Name"
                        - If user says "any role" or "all movies" -> use q="Actor Name" AND star="Actor Name"

                        IMPORTANT: For actor queries, ALWAYS include the actor name in BOTH the q parameter AND the star/star1 filter!

                        WHEN TO USE EACH TOOL:
                        
                        search_movies <Examples>:
                        - "Top rated movies" -> search_movies(q="movies", sort_by="IMDB_Rating", limit=10)
                        - "Best movies" -> search_movies(q="movies", sort_by="IMDB_Rating", limit=10)
                        - "Highest rated films" -> search_movies(q="movies", sort_by="IMDB_Rating", limit=10)
                        - "When did The Matrix release?" -> search_movies(q="The Matrix", limit=1)
                        - "Top 5 movies of 2019 by meta score" -> search_movies(q="movies", year=2019, sort_by="Meta_score", limit=5)
                        - "Top 7 comedy movies 2010-2020" -> search_movies(q="comedy", genre="comedy", year_start=2010, year_end=2020, sort_by="IMDB_Rating", limit=7)
                        - "Horror movies with meta > 85 and rating > 8" -> search_movies(q="horror", genre="horror", min_meta_score=85, min_rating=8)
                        - "Comedy movies about death" -> search_movies(q="death dying funeral", genre="comedy")
                        - "Movies about police before 1990" -> search_movies(q="police detective investigation", year_end=1989)
                        - "Al Pacino lead movies" -> search_movies(q="Al Pacino", star1="Al Pacino")
                        - "Al Pacino any role" -> search_movies(q="Al Pacino", star="Al Pacino")
                        - "Al Pacino movies rating > 8, grossed > $50M" -> search_movies(q="Al Pacino", star1="Al Pacino", min_rating=8, min_gross=50000000)

                        get_top_directors:
                        - "Top directors with $500M+ movies at least twice" -> get_top_directors(min_gross=500000000, min_count=2)

                        get_underrated_movies:
                        - "Top 10 movies with over 1M votes but lower gross" -> get_underrated_movies(min_votes=1000000, limit=10)

                        get_recommendations:
                        - After answering movie queries, offer similar movies
                        - Example:
                           - Provide overview of the movies : The Matrix, Inception ->
                             get_recommendations(avg_rating=8.5, avg_meta_score=85, exclude_titles="The Matrix,Inception")

                        RESPONSE FORMAT:
                        - Provide clear, direct answers based ONLY on tool results
                        - Include movie titles, years, ratings, gross EXACTLY as returned by the tool
                        - For plot summaries, use ONLY the overview field from the tool result
                        - NEVER add movies that were not in the tool's response
                        - If tool returns fewer results than user requested, explain that only X movies were found
                        
                        """


# ============ AGENT ============

movie_agent = Agent(
    name="Movie Agent",
    instructions=AGENT_INSTRUCTIONS,
    tools=[search_movies, 
           get_top_directors, 
           get_underrated_movies, 
           get_recommendations],    
    model_settings=ModelSettings(
        temperature=0.0,  # <-- set temperature here
        tool_choice="required",
            ),
    output_type=AgentOutput
)


# ============ RUNNER HELPER ============

def run_agent(conversation: list[TResponseInputItem], user_input: str):
    """
    Run the movie agent with a conversation and new user input.
    Returns (response_message, updated_conversation)
    """
    conversation.append({"role": "user", "content": user_input})
    result = Runner.run_sync(movie_agent, conversation)
    return result.final_output.Message, result.to_input_list()
