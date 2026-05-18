package types

import "time"

type CreateTaskRequest struct {
	Images   []CreateTaskImageRequest `json:"images"`
	Callback *TaskCallbackRequest     `json:"callback,omitempty"`
}

type CreateTaskImageRequest struct {
	ID  string `json:"id"`
	URL string `json:"url"`
}

type TaskCallbackRequest struct {
	URL  string `json:"url"`
	Mode string `json:"mode"`
}

type UploadedTaskImage struct {
	ID       string
	Filename string
	Content  []byte
}

type CreateTaskResponse struct {
	TaskID string `json:"task_id"`
	Status string `json:"status"`
}

type TaskImage struct {
	ID         string
	TaskID     string
	ExternalID string
	SourceURL  string
	Status     string
	Error      *string
}

type TaskDetails struct {
	ID              string            `json:"id"`
	Status          string            `json:"status"`
	TotalImages     int               `json:"total_images"`
	ProcessedImages int               `json:"processed_images"`
	Error           *string           `json:"error,omitempty"`
	Images          []TaskImageResult `json:"images"`
	Callback        *TaskCallbackInfo `json:"callback,omitempty"`
	CreatedAt       time.Time         `json:"created_at"`
	UpdatedAt       time.Time         `json:"updated_at"`
}

type TaskCallbackInfo struct {
	URL     string     `json:"url,omitempty"`
	Mode    string     `json:"mode"`
	Status  *string    `json:"status,omitempty"`
	Error   *string    `json:"error,omitempty"`
	SentAt  *time.Time `json:"sent_at,omitempty"`
	Enabled bool       `json:"enabled"`
}

type TaskImageResult struct {
	ID           string         `json:"id"`
	ExternalID   string         `json:"external_id"`
	SourceURL    string         `json:"source_url"`
	Status       string         `json:"status"`
	Error        *string        `json:"error,omitempty"`
	Numbers      []NumberResult `json:"numbers"`
	AnnotatedURL string         `json:"annotated_url,omitempty"`
}

type NumberResult struct {
	Number             string    `json:"number"`
	AvgDigitConfidence float64   `json:"avg_digit_confidence"`
	DetectConfidence   float64   `json:"detect_confidence"`
	SegmentConfidence  float64   `json:"segment_confidence"`
	CandidateScore     float64   `json:"candidate_score"`
	BBoxXYXY           []float64 `json:"bbox_xyxy"`
}
