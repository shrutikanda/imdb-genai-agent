import fs from 'fs';
import { parse } from 'csv-parse/sync';
import VectorStore from './vectorStore.js';
import dotenv from 'dotenv';

dotenv.config();

const CSV_PATH = process.argv[2] || './imdb_top_1000.csv';

async function indexMovies() {
    console.log(`Reading CSV from: ${CSV_PATH}`);
    
    // Read and parse CSV
    const csvContent = fs.readFileSync(CSV_PATH, 'utf-8');
    const records = parse(csvContent, {
        columns: true,
        skip_empty_lines: true
    });
    
    console.log(`Found ${records.length} movies to index`);
    
    // Initialize vector store
    const store = new VectorStore();
    
    // Index each movie (with rate limiting)
    let indexed = 0;
    const batchSize = 10;
    
    for (let i = 0; i < records.length; i += batchSize) {
        const batch = records.slice(i, i + batchSize);
        
        await Promise.all(batch.map(async (movie) => {
            try {
                await store.addMovie(movie);
                indexed++;
                process.stdout.write(`\rIndexed: ${indexed}/${records.length}`);
            } catch (error) {
                console.error(`\nError indexing ${movie.Series_Title}: ${error.message}`);
            }
        }));
        
        // Rate limit: wait 1 second between batches
        if (i + batchSize < records.length) {
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
    
    console.log('\n\nSaving embeddings to disk...');
    store.save();
    console.log('Done! Embeddings saved to embeddings.json');
}

indexMovies().catch(console.error);
