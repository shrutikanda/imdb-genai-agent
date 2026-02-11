import express from 'express';
import dotenv from 'dotenv';
import VectorStore from './vectorStore.js';

dotenv.config();

const app = express();
app.use(express.json());

// Initialize vector store and load embeddings
const movieStore = new VectorStore();
const hasEmbeddings = movieStore.load();

if (!hasEmbeddings) {
    console.warn('⚠️  No embeddings found. Run "npm run index -- <path-to-csv>" first.');
}

// ============ SINGLE HYBRID SEARCH ENDPOINT ============
app.get('/movies/search', async (req, res) => {
    const {
        q,                    // search query (required)
        limit = 10,           // max results
        genre,                // filter by genre
        year,                 // exact year
        year_start,           // year range start
        year_end,             // year range end
        min_rating,           // min IMDB rating
        min_meta_score,       // min meta score
        director,             // filter by director
        star,                 // filter by any actor (Star1-4)
        star1,                // filter by lead actor only
        min_votes,            // min votes
        min_gross,            // min gross earnings
        sort_by,              // sort field
        order = 'desc'        // sort order
    } = req.query;
    
    if (!q) {
        return res.status(400).json({ 
            error: 'Query parameter "q" is required',
            example: '/movies/search?q=police&genre=action&min_rating=7'
        });
    }
    
    if (movieStore.getCount() === 0) {
        return res.status(503).json({ error: 'Movies not loaded. Run indexing first.' });
    }
    
    try {
        const results = await movieStore.hybridSearch(q, {
            topK: parseInt(limit),
            genre,
            yearStart: year_start || year,
            yearEnd: year_end || year,
            minRating: min_rating,
            minMetaScore: min_meta_score,
            director,
            star,
            star1,
            minVotes: min_votes,
            minGross: min_gross,
            sortBy: sort_by,
            order
        });
        
        res.json({
            query: q,
            count: results.length,
            search_type: 'hybrid (BM25 + Vector + RRF)',
            results: results.map(r => ({
                title: r.movie.Series_Title,
                year: r.movie.Released_Year,
                genre: r.movie.Genre,
                rating: r.movie.IMDB_Rating,
                meta_score: r.movie.Meta_score,
                overview: r.movie.Overview,
                director: r.movie.Director,
                stars: [r.movie.Star1, r.movie.Star2, r.movie.Star3, r.movie.Star4].filter(Boolean),
                votes: r.movie.No_of_Votes,
                gross: r.movie.Gross,
                _scores: {
                    rrf: r.rrfScore?.toFixed(4),
                    bm25_rank: r.bm25Rank,
                    vector_rank: r.vectorRank
                }
            }))
        });
    } catch (error) {
        console.error('Search error:', error);
        res.status(500).json({ error: 'Search failed: ' + error.message });
    }
});

// Stats endpoint
app.get('/movies/stats', (req, res) => {
    res.json({
        totalMovies: movieStore.getCount(),
        indexed: movieStore.getCount() > 0,
        searchType: 'Hybrid (BM25 + Vector + RRF)',
        endpoints: [
            'GET /movies/search?q=<query>&filters...',
            'GET /movies/top-directors?min_gross=&min_count=',
            'GET /movies/underrated?min_votes=&limit='
        ]
    });
});

// ============ TOP DIRECTORS (Question 5) ============
app.get('/movies/top-directors', (req, res) => {
    const { min_gross = 500000000, min_count = 2 } = req.query;
    
    const movies = movieStore.getAllMovies();
    const directorStats = {};
    
    movies.forEach(movie => {
        const gross = parseInt(movie.Gross?.replace(/[$,]/g, '')) || 0;
        if (gross >= parseInt(min_gross)) {
            const dir = movie.Director;
            if (!directorStats[dir]) {
                directorStats[dir] = { movies: [], totalGross: 0 };
            }
            directorStats[dir].movies.push({
                title: movie.Series_Title,
                gross: movie.Gross,
                year: movie.Released_Year,
                grossNum: gross
            });
            directorStats[dir].totalGross += gross;
        }
    });
    
    const result = Object.entries(directorStats)
        .filter(([_, data]) => data.movies.length >= parseInt(min_count))
        .map(([director, data]) => ({
            director,
            highGrossMovies: data.movies.length,
            totalGross: `$${(data.totalGross / 1000000).toFixed(0)}M`,
            movies: data.movies.sort((a, b) => b.grossNum - a.grossNum)
        }))
        .sort((a, b) => b.highGrossMovies - a.highGrossMovies);
    
    res.json({ 
        query: `Directors with ${min_count}+ movies grossing $${min_gross/1000000}M+`,
        count: result.length, 
        results: result 
    });
});

// ============ UNDERRATED MOVIES (Question 6) ============
app.get('/movies/underrated', (req, res) => {
    const { min_votes = 1000000, limit = 10 } = req.query;
    
    const movies = movieStore.getAllMovies()
        .filter(m => {
            const votes = parseInt(m.No_of_Votes?.replace(/,/g, '')) || 0;
            return votes >= parseInt(min_votes);
        })
        .sort((a, b) => {
            const grossA = parseInt(a.Gross?.replace(/[$,]/g, '')) || 0;
            const grossB = parseInt(b.Gross?.replace(/[$,]/g, '')) || 0;
            return grossA - grossB;
        })
        .slice(0, parseInt(limit));
    
    res.json({
        query: `Movies with ${min_votes}+ votes, sorted by lowest gross`,
        count: movies.length,
        results: movies.map(m => ({
            title: m.Series_Title,
            year: m.Released_Year,
            votes: m.No_of_Votes,
            gross: m.Gross || 'N/A',
            rating: m.IMDB_Rating
        }))
    });
});

// ============ RECOMMENDATIONS ENDPOINT ============
app.get('/movies/recommendations', (req, res) => {
    const { avg_rating = 8.0, avg_meta_score = 80, exclude = '', limit = 5 } = req.query;
    
    const excludeList = exclude ? exclude.split(',').map(t => t.trim().toLowerCase()) : [];
    const targetRating = parseFloat(avg_rating);
    const targetMeta = parseFloat(avg_meta_score);
    
    const scored = movieStore.getAllMovies()
        .filter(m => !excludeList.includes(m.Series_Title.toLowerCase()))
        .map(m => {
            const rating = parseFloat(m.IMDB_Rating) || 0;
            const meta = parseFloat(m.Meta_score) || 0;
            
            // Calculate similarity score (lower is more similar)
            const ratingDiff = Math.abs(rating - targetRating);
            const metaDiff = Math.abs(meta - targetMeta) / 10; // Normalize
            const similarity = ratingDiff + metaDiff;
            
            return { movie: m, similarity };
        })
        .sort((a, b) => a.similarity - b.similarity)
        .slice(0, parseInt(limit));
    
    res.json({
        based_on: { target_rating: targetRating, target_meta_score: targetMeta },
        count: scored.length,
        recommendations: scored.map(s => ({
            title: s.movie.Series_Title,
            year: s.movie.Released_Year,
            rating: s.movie.IMDB_Rating,
            meta_score: s.movie.Meta_score,
            genre: s.movie.Genre,
            similarity: s.similarity.toFixed(2)
        }))
    });
});

app.listen(3000, () => {
    console.log('🎬 IMDB Hybrid Search API running on http://localhost:3000');
    console.log(`📊 Loaded ${movieStore.getCount()} movies`);
    console.log('\n📌 Endpoints:');
    console.log('  GET /movies/search?q=<query>&genre=&year=&min_rating=...');
    console.log('  GET /movies/top-directors?min_gross=500000000&min_count=2');
    console.log('  GET /movies/underrated?min_votes=1000000&limit=10');
    console.log('  GET /movies/recommendations?avg_rating=8&avg_meta_score=80');
});