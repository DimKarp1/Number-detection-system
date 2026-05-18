package ml

type RecognizeResponse struct {
	RequestID         string             `json:"request_id"`
	RecognizedNumbers []RecognizedNumber `json:"recognized_numbers"`
	Candidates        []Candidate        `json:"candidates,omitempty"`
	DebugDir          string             `json:"debug_dir,omitempty"`
}

type RecognizedNumber struct {
	Number                   string               `json:"number"`
	AvgDigitConfidence       float64              `json:"avg_digit_confidence"`
	DetectConfidence         float64              `json:"detect_confidence"`
	SegmentConfidence        float64              `json:"segment_confidence"`
	Orientation              string               `json:"orientation"`
	OrientationAttempts      []OrientationAttempt `json:"orientation_attempts"`
	PlateRotatedToHorizontal bool                 `json:"plate_rotated_to_horizontal"`
	PlateAspect              float64              `json:"plate_aspect"`
	PlateSize                []int                `json:"plate_size"`
	BBoxXYXY                 []float64            `json:"bbox_xyxy"`
	CandidateScore           float64              `json:"candidate_score"`
	DigitSpanRatio           float64              `json:"digit_span_ratio"`
}

type OrientationAttempt struct {
	Orientation        string  `json:"orientation"`
	Number             string  `json:"number"`
	Score              float64 `json:"score"`
	DigitsCount        int     `json:"digits_count"`
	AvgDigitConfidence float64 `json:"avg_digit_confidence"`
}

type Candidate struct {
	Rank                     int                  `json:"rank"`
	DetectConfidence         float64              `json:"detect_confidence"`
	DetectBBoxXYXY           []float64            `json:"detect_bbox_xyxy"`
	ExpandedCropXYXY         []int                `json:"expanded_crop_xyxy"`
	SegmentFound             bool                 `json:"segment_found"`
	SegmentConfidence        *float64             `json:"segment_confidence"`
	Number                   string               `json:"number"`
	Digits                   []Digit              `json:"digits"`
	Orientation              string               `json:"orientation"`
	OrientationAttempts      []OrientationAttempt `json:"orientation_attempts"`
	PlateRotatedToHorizontal bool                 `json:"plate_rotated_to_horizontal"`
	PlateAspect              float64              `json:"plate_aspect"`
	PlateSize                []int                `json:"plate_size"`
	DigitSequenceScore       float64              `json:"digit_sequence_score"`
	AvgDigitConfidence       float64              `json:"avg_digit_confidence"`
	CandidateScore           float64              `json:"candidate_score"`
	SkipReason               *string              `json:"skip_reason"`
}

type Digit struct {
	Digit      string    `json:"digit"`
	Confidence float64   `json:"confidence"`
	BBoxXYXY   []float64 `json:"bbox_xyxy"`
}
