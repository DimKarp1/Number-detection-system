CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    status TEXT NOT NULL DEFAULT 'created',
    total_images INT NOT NULL DEFAULT 0,
    processed_images INT NOT NULL DEFAULT 0,
    error TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS task_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    external_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    error TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS image_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_image_id UUID NOT NULL REFERENCES task_images(id) ON DELETE CASCADE,
    number TEXT NOT NULL,
    avg_digit_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    detect_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    segment_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    candidate_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    bbox_xyxy JSONB NOT NULL DEFAULT '[]'::jsonb,
    raw_response JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_task_images_task_id ON task_images(task_id);
CREATE INDEX IF NOT EXISTS idx_image_results_task_image_id ON image_results(task_image_id);
CREATE INDEX IF NOT EXISTS idx_image_results_number ON image_results(number);
