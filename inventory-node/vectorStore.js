import OpenAI from 'openai';
import MiniSearch from 'minisearch';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const EMBEDDINGS_FILE = path.join(__dirname, 'embeddings.json');

class VectorStore {
    constructor() {
        this.openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
        this.embeddings = [];
        this.movies = [];
        
        // BM25-style keyword search
        this.miniSearch = new MiniSearch({
            fields: ['Series_Title', 'Genre', 'Overview', 'Director', 'Star1', 'Star2', 'Star3', 'Star4'],
            storeFields: ['Series_Title'],
            idField: 'id',
            searchOptions: {
                boost: { Series_Title: 3, Overview: 2, Genre: 1.5, Director: 1.5 },
                fuzzy: 0.2, //typo tolerance - 20% character difference
                prefix: true // "Spiel" matches "Spielberg"
            }
        });
    }

    // Load existing embeddings from file
    load() {
        if (fs.existsSync(EMBEDDINGS_FILE)) {
            const data = JSON.parse(fs.readFileSync(EMBEDDINGS_FILE, 'utf-8'));
            this.embeddings = data.embeddings || [];
            this.movies = data.movies || [];
            
            // Build MiniSearch index
            const docs = this.movies.map((m, idx) => ({ ...m, id: idx }));
            this.miniSearch.addAll(docs);
            
            console.log(`Loaded ${this.movies.length} movies from cache`);
            console.log(`Built BM25 index for hybrid search`);
            return true;
        }
        return false;
    }

    // Save embeddings to file
    save() {
        const data = { embeddings: this.embeddings, movies: this.movies };
        fs.writeFileSync(EMBEDDINGS_FILE, JSON.stringify(data));
        console.log(`Saved ${this.movies.length} movies to cache`);
    }

    // Generate embedding for text
    async getEmbedding(text) {
        const response = await this.openai.embeddings.create({
            model: 'text-embedding-3-small',
            input: text,
        });
        return response.data[0].embedding;
    }

    // Add movie with its embedding
    async addMovie(movie) {
        const searchText = `Movies titles ${movie.Series_Title}. ${movie.Genre}. ${movie.Overview}. Directed by ${movie.Director}. Starring ${movie.Star1}, ${movie.Star2}.`;
        
        const embedding = await this.getEmbedding(searchText);
        this.embeddings.push(embedding);
        this.movies.push(movie);
    }

    // Cosine similarity between two vectors    
    cosineSimilarity(vecA, vecB) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        
        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    // Vector-only search
    async vectorSearch(query, topK = 20) {
        const queryEmbedding = await this.getEmbedding(query);
        
        const results = this.movies.map((movie, index) => ({
            movie,
            index,
            score: this.cosineSimilarity(queryEmbedding, this.embeddings[index])
        }));
        
        results.sort((a, b) => b.score - a.score);
        return results.slice(0, topK);
    }

    // BM25 keyword search
    bm25Search(query, topK = 20) {
        const results = this.miniSearch.search(query, { limit: topK });
        return results.map(r => ({
            movie: this.movies[r.id],
            index: r.id,
            score: r.score
        }));
    }

    // Reciprocal Rank Fusion (RRF)
    reciprocalRankFusion(bm25Results, vectorResults, k = 60) {
        const scores = new Map();
        
        // Score from BM25
        bm25Results.forEach((r, rank) => {
            const key = r.index;
            scores.set(key, (scores.get(key) || 0) + 1 / (k + rank + 1));
        });
        
        // Score from Vector
        vectorResults.forEach((r, rank) => {
            const key = r.index;
            scores.set(key, (scores.get(key) || 0) + 1 / (k + rank + 1));
        });
        
        // Convert to array and sort
        const combined = [...scores.entries()]
            .map(([index, score]) => ({
                movie: this.movies[index],
                index,
                rrfScore: score,
                bm25Rank: bm25Results.findIndex(r => r.index === index) + 1 || null,
                vectorRank: vectorResults.findIndex(r => r.index === index) + 1 || null
            }))
            .sort((a, b) => b.rrfScore - a.rrfScore);
        
        return combined;
    }

    // HYBRID SEARCH: BM25 + Vector + RRF
    async hybridSearch(query, options = {}) {
        const {
            topK = 10,
            genre = null,
            yearStart = null,
            yearEnd = null,
            minRating = null,
            minMetaScore = null,
            director = null,
            star = null,
            star1 = null,
            minVotes = null,
            minGross = null,
            sortBy = null,
            order = 'desc'
        } = options;

        const filters = {
            genre, yearStart, yearEnd, minRating, minMetaScore,
            director, star, star1, minVotes, minGross
        };

        // Check if query is generic (user wants filtered results, not semantic search)
        const isGenericQuery = !query || 
            query.trim() === '' || 
            /^(movies?|films?|all|top|best|highest|show|list|get|find)$/i.test(query.trim());

        const hasFilters = Object.values(filters).some(v => v !== null && v !== undefined);

        // MODE 1: Filter-only (generic query + filters present)
        if (isGenericQuery && hasFilters) {
            console.log('MODE: Filter-only (skipping semantic search)');
            
            let results = this.movies.map((movie, index) => ({
                movie,
                index,
                rrfScore: 1,
                bm25Rank: null,
                vectorRank: null
            }));

            // Apply filters to ALL movies
            results = this.applyFilters(results, filters);
            console.log(`After filters: ${results.length} movies match criteria`);

            // Sort by specified field or default to IMDB_Rating
            const effectiveSortBy = sortBy || 'IMDB_Rating';
            results = this.sortResults(results, effectiveSortBy, order);

            return results.slice(0, topK);
        }

        // MODE 2: Semantic search (specific query)
        console.log('MODE: Hybrid semantic search (BM25 + Vector + RRF)');

        // 1. Get BM25 results
        const bm25Results = this.bm25Search(query, topK * 3);
        console.log('bm25Results:', bm25Results.map(r => `${r.movie.Series_Title} (score: ${r.score.toFixed(2)})`));

        // 2. Get Vector results
        const vectorResults = await this.vectorSearch(query, topK * 3);
        console.log('vectorResults:', vectorResults.map(r => `${r.movie.Series_Title} (score: ${r.score.toFixed(2)})`));

        // 3. Combine with RRF
        let results = this.reciprocalRankFusion(bm25Results, vectorResults);
        console.log('RRFResults:', results.map(r => `${r.movie.Series_Title} (score: ${r.rrfScore.toFixed(2)})`));

        // 4. Apply filters (if any)
        if (hasFilters) {
            results = this.applyFilters(results, filters);
            console.log('After filters:', results.map(r => `${r.movie.Series_Title} (score: ${r.rrfScore.toFixed(2)})`));
        }

        // 5. Re-sort if specified
        if (sortBy) {
            results = this.sortResults(results, sortBy, order);
        }

        return results.slice(0, topK);
    }

    // Apply structured filters
    applyFilters(results, filters) {
        return results.filter(r => {
            const m = r.movie;
            
            if (filters.genre && !m.Genre.toLowerCase().includes(filters.genre.toLowerCase())) {
                return false;
            }
            if (filters.yearStart && parseInt(m.Released_Year) < parseInt(filters.yearStart)) {
                return false;
            }
            if (filters.yearEnd && parseInt(m.Released_Year) > parseInt(filters.yearEnd)) {
                return false;
            }
            if (filters.minRating && parseFloat(m.IMDB_Rating) < parseFloat(filters.minRating)) {
                return false;
            }
            if (filters.minMetaScore && parseInt(m.Meta_score) < parseInt(filters.minMetaScore)) {
                return false;
            }
            if (filters.director && !m.Director.toLowerCase().includes(filters.director.toLowerCase())) {
                return false;
            }
            if (filters.star) {
                const starLower = filters.star.toLowerCase();
                const hasActor = [m.Star1, m.Star2, m.Star3, m.Star4]
                    .some(s => s && s.toLowerCase().includes(starLower));
                if (!hasActor) return false;
            }
            if (filters.star1 && !m.Star1.toLowerCase().includes(filters.star1.toLowerCase())) {
                return false;
            }
            if (filters.minVotes) {
                const votes = parseInt(m.No_of_Votes?.replace(/,/g, '')) || 0;
                if (votes < parseInt(filters.minVotes)) return false;
            }
            if (filters.minGross) {
                const gross = parseInt(m.Gross?.replace(/[$,]/g, '')) || 0;
                if (gross < parseInt(filters.minGross)) return false;
            }
            
            return true;
        });
    }

    // Sort by field
    sortResults(results, sortBy, order) {
        return results.sort((a, b) => {
            let valA, valB;
            const mA = a.movie;
            const mB = b.movie;
            
            switch (sortBy) {
                case 'IMDB_Rating':
                    valA = parseFloat(mA.IMDB_Rating) || 0;
                    valB = parseFloat(mB.IMDB_Rating) || 0;
                    break;
                case 'Meta_score':
                    valA = parseInt(mA.Meta_score) || 0;
                    valB = parseInt(mB.Meta_score) || 0;
                    break;
                case 'Gross':
                    valA = parseInt(mA.Gross?.replace(/[$,]/g, '')) || 0;
                    valB = parseInt(mB.Gross?.replace(/[$,]/g, '')) || 0;
                    break;
                case 'Released_Year':
                    valA = parseInt(mA.Released_Year) || 0;
                    valB = parseInt(mB.Released_Year) || 0;
                    break;
                case 'No_of_Votes':
                    valA = parseInt(mA.No_of_Votes?.replace(/,/g, '')) || 0;
                    valB = parseInt(mB.No_of_Votes?.replace(/,/g, '')) || 0;
                    break;
                default:
                    return 0;
            }
            
            return order === 'asc' ? valA - valB : valB - valA;
        });
    }

    // Legacy search method for backwards compatibility
    async search(query, topK = 5) {
        return this.vectorSearch(query, topK);
    }

    // Get all movies (for filtering/aggregation)
    getAllMovies() {
        return this.movies;
    }   

    // Get total movie count
    getCount() {
        return this.movies.length;
    }
}

export default VectorStore;
