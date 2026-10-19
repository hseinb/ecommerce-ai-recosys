-- Enable pgvector extension in Supabase
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Core Catalog & Inventory State
CREATE TABLE IF NOT EXISTS items (
    item_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category_id VARCHAR(100),
    category_name VARCHAR(255),
    unit_cost NUMERIC(12, 4) NOT NULL DEFAULT 0.0,
    retail_price NUMERIC(12, 4) NOT NULL DEFAULT 0.0,
    margin NUMERIC(8, 4) GENERATED ALWAYS AS (
        CASE WHEN retail_price > 0 THEN (retail_price - unit_cost) / retail_price ELSE 0 END
    ) STORED,
    current_stock INT NOT NULL DEFAULT 0,
    supplier_lead_time_days INT NOT NULL DEFAULT 7,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Spatial Representation: Item Embeddings (Vector Latent Space)
CREATE TABLE IF NOT EXISTS item_embeddings (
    item_id VARCHAR(100) PRIMARY KEY REFERENCES items(item_id) ON DELETE CASCADE,
    embedding vector(64) NOT NULL,
    cluster_id INT DEFAULT 0,
    version VARCHAR(50) DEFAULT 'v1.0',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW Vector Index for fast Cosine Distance retrieval on Items
CREATE INDEX IF NOT EXISTS idx_item_embeddings_hnsw 
ON item_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 3. Spatial Representation: Customer Embeddings (User Vector Latent Space)
CREATE TABLE IF NOT EXISTS customer_embeddings (
    customer_id VARCHAR(100) PRIMARY KEY,
    embedding vector(64) NOT NULL,
    persona_segment_id INT DEFAULT 0,
    version VARCHAR(50) DEFAULT 'v1.0',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW Vector Index for fast Cosine Distance retrieval on Customers (Lookalike Search)
CREATE INDEX IF NOT EXISTS idx_customer_embeddings_hnsw 
ON customer_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 4. Recommendations: Item-to-Item Similarity Cache
CREATE TABLE IF NOT EXISTS item_similarities (
    id BIGSERIAL PRIMARY KEY,
    source_item_id VARCHAR(100) REFERENCES items(item_id) ON DELETE CASCADE,
    target_item_id VARCHAR(100) REFERENCES items(item_id) ON DELETE CASCADE,
    similarity_score FLOAT NOT NULL,
    rank INT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_item_pair UNIQUE (source_item_id, target_item_id)
);

CREATE INDEX IF NOT EXISTS idx_item_similarities_source ON item_similarities(source_item_id, rank);

-- 5. Recommendations: Customer Lookalike Peers
CREATE TABLE IF NOT EXISTS customer_lookalikes (
    id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(100) NOT NULL,
    lookalike_customer_id VARCHAR(100) NOT NULL,
    similarity_score FLOAT NOT NULL,
    rank INT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_customer_peer UNIQUE (customer_id, lookalike_customer_id)
);

CREATE INDEX IF NOT EXISTS idx_customer_lookalikes_cust ON customer_lookalikes(customer_id);

-- 6. Recommendations: Peer Collaborative Recommendations (One purchase from similar customer -> recommended)
CREATE TABLE IF NOT EXISTS customer_peer_recommendations (
    id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(100) NOT NULL,
    item_id VARCHAR(100) REFERENCES items(item_id) ON DELETE CASCADE,
    collaborative_score FLOAT NOT NULL,
    recommended_via_lookalike_ids TEXT[],
    rank INT NOT NULL,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_customer_peer_rec UNIQUE (customer_id, rank)
);

CREATE INDEX IF NOT EXISTS idx_customer_peer_recs_cust ON customer_peer_recommendations(customer_id);

-- 7. Recommendations: Customer Direct Personalized Top-K
CREATE TABLE IF NOT EXISTS customer_recommendations (
    id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(100) NOT NULL,
    item_id VARCHAR(100) REFERENCES items(item_id) ON DELETE CASCADE,
    predicted_score FLOAT NOT NULL,
    rank INT NOT NULL,
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_customer_item_rank UNIQUE (customer_id, rank)
);

CREATE INDEX IF NOT EXISTS idx_customer_recs_cust ON customer_recommendations(customer_id);

-- 8. Seasonal & Featured Rankings
CREATE TABLE IF NOT EXISTS seasonal_featured_items (
    id BIGSERIAL PRIMARY KEY,
    week_of_year INT NOT NULL,
    quarter INT NOT NULL,
    item_id VARCHAR(100) REFERENCES items(item_id) ON DELETE CASCADE,
    seasonal_velocity_index FLOAT NOT NULL,
    boost_rank INT NOT NULL,
    year INT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_seasonal_item UNIQUE (year, week_of_year, item_id)
);

-- 9. Inventory & Pricing Decision Layer
CREATE TABLE IF NOT EXISTS inventory_pricing_decisions (
    item_id VARCHAR(100) PRIMARY KEY REFERENCES items(item_id) ON DELETE CASCADE,
    cluster_id INT DEFAULT 0,
    avg_daily_demand FLOAT NOT NULL DEFAULT 0.0,
    demand_std_dev FLOAT NOT NULL DEFAULT 0.0,
    lead_time_days FLOAT NOT NULL DEFAULT 7.0,
    safety_stock FLOAT NOT NULL DEFAULT 0.0,
    reorder_point FLOAT NOT NULL DEFAULT 0.0,
    stock_status VARCHAR(50) DEFAULT 'OK', -- 'REORDER_NOW', 'SAFETY_BREACH', 'OK', 'OVERSTOCKED'
    price_elasticity_ped FLOAT,
    recommended_price_action VARCHAR(50) DEFAULT 'MAINTAIN', -- 'MARKDOWN_PROMOTE', 'MARGIN_HIKE', 'MAINTAIN'
    optimal_price NUMERIC(12, 4),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 10. Visual Lens Index: Precomputed Cloudflare R2 Image Embeddings & Item Text Embeddings
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS public.recosys_visual_index (
    item_id INTEGER PRIMARY KEY REFERENCES public.items(item_id) ON DELETE CASCADE,
    code TEXT,
    item_name TEXT NOT NULL,
    category_name TEXT,
    description TEXT,
    material TEXT,
    house_rooms TEXT,
    tags TEXT,
    r2_thumb_url TEXT,
    r2_view_url TEXT,
    searchable_text TEXT,
    image_embedding vector(768),
    text_embedding vector(768),
    image_source_updated_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_recosys_visual_image_hnsw
ON public.recosys_visual_index USING hnsw (image_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_recosys_visual_text_hnsw
ON public.recosys_visual_index USING hnsw (text_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_recosys_visual_trgm
ON public.recosys_visual_index USING gin (searchable_text gin_trgm_ops);

-- 11. Visual Lens Search History & Comparison Log
CREATE TABLE IF NOT EXISTS public.recosys_lens_searches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_label TEXT,
    extracted_keywords TEXT[],
    visual_description_en TEXT,
    visual_description_ar TEXT,
    detected_bbox INTEGER[],
    top_image_matches JSONB DEFAULT '[]'::jsonb,
    top_keyword_matches JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT timezone('utc'::text, now())
);

