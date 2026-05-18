package repo

import (
	"context"
	"encoding/json"
	"fmt"

	"yolo12-number-system/backend/internal/types"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"
)

type TaskRepo struct {
	pool *pgxpool.Pool
}

func NewTaskRepo(pool *pgxpool.Pool) *TaskRepo {
	return &TaskRepo{
		pool: pool,
	}
}

func (r *TaskRepo) CreateTask(
	ctx context.Context,
	images []types.CreateTaskImageRequest,
	callback *types.TaskCallbackRequest,
) (string, []types.TaskImage, error) {
	tx, err := r.pool.Begin(ctx)
	if err != nil {
		return "", nil, fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	callbackURL := ""
	callbackMode := "none"

	if callback != nil && callback.URL != "" {
		callbackURL = callback.URL
		callbackMode = callback.Mode

		if callbackMode == "" {
			callbackMode = "result"
		}
	}

	var taskID string

	err = tx.QueryRow(
		ctx,
		`
		INSERT INTO tasks (
			status,
			total_images,
			processed_images,
			callback_url,
			callback_mode
		)
		VALUES ('created', $1, 0, $2, $3)
		RETURNING id::text
		`,
		len(images),
		nullIfEmpty(callbackURL),
		callbackMode,
	).Scan(&taskID)
	if err != nil {
		return "", nil, fmt.Errorf("insert task: %w", err)
	}

	taskImages := make([]types.TaskImage, 0, len(images))

	for _, image := range images {
		var taskImageID string

		err := tx.QueryRow(
			ctx,
			`
			INSERT INTO task_images (task_id, external_id, source_url, status)
			VALUES ($1, $2, $3, 'created')
			RETURNING id::text
			`,
			taskID,
			image.ID,
			image.URL,
		).Scan(&taskImageID)
		if err != nil {
			return "", nil, fmt.Errorf("insert task image: %w", err)
		}

		taskImages = append(taskImages, types.TaskImage{
			ID:         taskImageID,
			TaskID:     taskID,
			ExternalID: image.ID,
			SourceURL:  image.URL,
			Status:     "created",
		})
	}

	if err := tx.Commit(ctx); err != nil {
		return "", nil, fmt.Errorf("commit tx: %w", err)
	}

	return taskID, taskImages, nil
}

func (r *TaskRepo) SetTaskStatus(
	ctx context.Context,
	taskID string,
	status string,
	errorText *string,
) error {
	_, err := r.pool.Exec(
		ctx,
		`
		UPDATE tasks
		SET status = $2,
		    error = $3,
		    updated_at = now()
		WHERE id = $1
		`,
		taskID,
		status,
		errorText,
	)
	if err != nil {
		return fmt.Errorf("update task status: %w", err)
	}

	return nil
}

func (r *TaskRepo) SetTaskImageStatus(
	ctx context.Context,
	taskImageID string,
	status string,
	errorText *string,
) error {
	_, err := r.pool.Exec(
		ctx,
		`
		UPDATE task_images
		SET status = $2,
		    error = $3,
		    updated_at = now()
		WHERE id = $1
		`,
		taskImageID,
		status,
		errorText,
	)
	if err != nil {
		return fmt.Errorf("update task image status: %w", err)
	}

	return nil
}

func (r *TaskRepo) SaveImageResults(
	ctx context.Context,
	taskImageID string,
	numbers []types.NumberResult,
	rawResponse []byte,
) error {
	tx, err := r.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	_, err = tx.Exec(
		ctx,
		`
		DELETE FROM image_results
		WHERE task_image_id = $1
		`,
		taskImageID,
	)
	if err != nil {
		return fmt.Errorf("delete old image results: %w", err)
	}

	if len(rawResponse) == 0 {
		rawResponse = []byte(`{}`)
	}

	for _, number := range numbers {
		bboxRaw, err := json.Marshal(number.BBoxXYXY)
		if err != nil {
			return fmt.Errorf("marshal bbox: %w", err)
		}

		_, err = tx.Exec(
			ctx,
			`
			INSERT INTO image_results (
				task_image_id,
				number,
				avg_digit_confidence,
				detect_confidence,
				segment_confidence,
				candidate_score,
				bbox_xyxy,
				raw_response
			)
			VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb)
			`,
			taskImageID,
			number.Number,
			number.AvgDigitConfidence,
			number.DetectConfidence,
			number.SegmentConfidence,
			number.CandidateScore,
			string(bboxRaw),
			string(rawResponse),
		)
		if err != nil {
			return fmt.Errorf("insert image result: %w", err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit tx: %w", err)
	}

	return nil
}

func (r *TaskRepo) RefreshTaskProgress(ctx context.Context, taskID string) error {
	var totalImages int
	var processedImages int
	var failedImages int

	err := r.pool.QueryRow(
		ctx,
		`
		SELECT
			t.total_images,
			COUNT(ti.id) FILTER (WHERE ti.status IN ('completed', 'failed'))::int,
			COUNT(ti.id) FILTER (WHERE ti.status = 'failed')::int
		FROM tasks t
		LEFT JOIN task_images ti ON ti.task_id = t.id
		WHERE t.id = $1
		GROUP BY t.id
		`,
		taskID,
	).Scan(&totalImages, &processedImages, &failedImages)
	if err != nil {
		return fmt.Errorf("calculate task progress: %w", err)
	}

	status := "processing"
	var errorText *string

	if processedImages >= totalImages {
		status = "completed"

		if failedImages > 0 {
			status = "completed_with_errors"
			msg := fmt.Sprintf("%d image(s) failed", failedImages)
			errorText = &msg
		}
	}

	_, err = r.pool.Exec(
		ctx,
		`
		UPDATE tasks
		SET status = $2,
		    processed_images = $3,
		    error = $4,
		    updated_at = now()
		WHERE id = $1
		`,
		taskID,
		status,
		processedImages,
		errorText,
	)
	if err != nil {
		return fmt.Errorf("update task progress: %w", err)
	}

	return nil
}

func (r *TaskRepo) SetCallbackResult(
	ctx context.Context,
	taskID string,
	status string,
	errorText *string,
) error {
	_, err := r.pool.Exec(
		ctx,
		`
		UPDATE tasks
		SET callback_status = $2,
		    callback_error = $3,
		    callback_sent_at = now(),
		    updated_at = now()
		WHERE id = $1
		`,
		taskID,
		status,
		errorText,
	)
	if err != nil {
		return fmt.Errorf("update callback result: %w", err)
	}

	return nil
}

func (r *TaskRepo) GetTaskDetails(ctx context.Context, taskID string) (types.TaskDetails, error) {
	var details types.TaskDetails
	var taskError pgtype.Text

	var callbackURL pgtype.Text
	var callbackMode string
	var callbackStatus pgtype.Text
	var callbackError pgtype.Text
	var callbackSentAt pgtype.Timestamptz

	err := r.pool.QueryRow(
		ctx,
		`
		SELECT
			id::text,
			status,
			total_images,
			processed_images,
			error,
			callback_url,
			callback_mode,
			callback_status,
			callback_error,
			callback_sent_at,
			created_at,
			updated_at
		FROM tasks
		WHERE id = $1
		`,
		taskID,
	).Scan(
		&details.ID,
		&details.Status,
		&details.TotalImages,
		&details.ProcessedImages,
		&taskError,
		&callbackURL,
		&callbackMode,
		&callbackStatus,
		&callbackError,
		&callbackSentAt,
		&details.CreatedAt,
		&details.UpdatedAt,
	)
	if err != nil {
		if err == pgx.ErrNoRows {
			return types.TaskDetails{}, ErrNotFound
		}

		return types.TaskDetails{}, fmt.Errorf("select task: %w", err)
	}

	if taskError.Valid {
		details.Error = &taskError.String
	}

	if callbackURL.Valid && callbackURL.String != "" {
		callbackInfo := &types.TaskCallbackInfo{
			URL:     callbackURL.String,
			Mode:    callbackMode,
			Enabled: true,
		}

		if callbackStatus.Valid {
			callbackInfo.Status = &callbackStatus.String
		}

		if callbackError.Valid {
			callbackInfo.Error = &callbackError.String
		}

		if callbackSentAt.Valid {
			sentAt := callbackSentAt.Time
			callbackInfo.SentAt = &sentAt
		}

		details.Callback = callbackInfo
	}

	rows, err := r.pool.Query(
		ctx,
		`
		SELECT
			id::text,
			external_id,
			source_url,
			status,
			error
		FROM task_images
		WHERE task_id = $1
		ORDER BY created_at ASC
		`,
		taskID,
	)
	if err != nil {
		return types.TaskDetails{}, fmt.Errorf("select task images: %w", err)
	}
	defer rows.Close()

	images := make([]types.TaskImageResult, 0)

	for rows.Next() {
		var image types.TaskImageResult
		var imageError pgtype.Text

		err := rows.Scan(
			&image.ID,
			&image.ExternalID,
			&image.SourceURL,
			&image.Status,
			&imageError,
		)
		if err != nil {
			return types.TaskDetails{}, fmt.Errorf("scan task image: %w", err)
		}

		if imageError.Valid {
			image.Error = &imageError.String
		}

		numbers, err := r.getImageResults(ctx, image.ID)
		if err != nil {
			return types.TaskDetails{}, err
		}

		image.Numbers = numbers

		images = append(images, image)
	}

	if err := rows.Err(); err != nil {
		return types.TaskDetails{}, fmt.Errorf("iterate task images: %w", err)
	}

	details.Images = images

	return details, nil
}

func (r *TaskRepo) getImageResults(ctx context.Context, taskImageID string) ([]types.NumberResult, error) {
	rows, err := r.pool.Query(
		ctx,
		`
		SELECT
			number,
			avg_digit_confidence,
			detect_confidence,
			segment_confidence,
			candidate_score,
			bbox_xyxy
		FROM image_results
		WHERE task_image_id = $1
		ORDER BY candidate_score DESC
		`,
		taskImageID,
	)
	if err != nil {
		return nil, fmt.Errorf("select image results: %w", err)
	}
	defer rows.Close()

	results := make([]types.NumberResult, 0)

	for rows.Next() {
		var result types.NumberResult
		var bboxRaw []byte

		err := rows.Scan(
			&result.Number,
			&result.AvgDigitConfidence,
			&result.DetectConfidence,
			&result.SegmentConfidence,
			&result.CandidateScore,
			&bboxRaw,
		)
		if err != nil {
			return nil, fmt.Errorf("scan image result: %w", err)
		}

		if len(bboxRaw) > 0 {
			_ = json.Unmarshal(bboxRaw, &result.BBoxXYXY)
		}

		results = append(results, result)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate image results: %w", err)
	}

	return results, nil
}

func (r *TaskRepo) GetTaskImageResult(
	ctx context.Context,
	taskImageID string,
) (types.TaskImageResult, error) {
	var image types.TaskImageResult
	var imageError pgtype.Text

	err := r.pool.QueryRow(
		ctx,
		`
		SELECT
			id::text,
			external_id,
			source_url,
			status,
			error
		FROM task_images
		WHERE id = $1
		`,
		taskImageID,
	).Scan(
		&image.ID,
		&image.ExternalID,
		&image.SourceURL,
		&image.Status,
		&imageError,
	)
	if err != nil {
		if err == pgx.ErrNoRows {
			return types.TaskImageResult{}, ErrNotFound
		}

		return types.TaskImageResult{}, fmt.Errorf("select task image: %w", err)
	}

	if imageError.Valid {
		image.Error = &imageError.String
	}

	numbers, err := r.getImageResults(ctx, image.ID)
	if err != nil {
		return types.TaskImageResult{}, err
	}

	image.Numbers = numbers

	return image, nil
}

func nullIfEmpty(value string) any {
	if value == "" {
		return nil
	}

	return value
}

var ErrNotFound = fmt.Errorf("not found")
