package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	imagedraw "image/draw"
	"image/jpeg"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"yolo12-number-system/backend/internal/config"
	"yolo12-number-system/backend/internal/ml"
	"yolo12-number-system/backend/internal/repo"
	"yolo12-number-system/backend/internal/types"

	"github.com/google/uuid"
)

type TaskService struct {
	repo       *repo.TaskRepo
	mlClient   *ml.Client
	cfg        config.Config
	httpClient *http.Client
}

func NewTaskService(
	repo *repo.TaskRepo,
	mlClient *ml.Client,
	cfg config.Config,
) *TaskService {
	return &TaskService{
		repo:     repo,
		mlClient: mlClient,
		cfg:      cfg,
		httpClient: &http.Client{
			Timeout: 90 * time.Second,
		},
	}
}

func (s *TaskService) CreateTask(
	ctx context.Context,
	req types.CreateTaskRequest,
) (types.CreateTaskResponse, error) {
	if len(req.Images) == 0 {
		return types.CreateTaskResponse{}, fmt.Errorf("images list is empty")
	}

	for i := range req.Images {
		req.Images[i].ID = strings.TrimSpace(req.Images[i].ID)
		req.Images[i].URL = strings.TrimSpace(req.Images[i].URL)

		if req.Images[i].URL == "" {
			return types.CreateTaskResponse{}, fmt.Errorf("image url is required")
		}

		if req.Images[i].ID == "" {
			req.Images[i].ID = fmt.Sprintf("image_%d", i+1)
		}
	}

	callback := normalizeCallback(req.Callback)

	taskID, taskImages, err := s.repo.CreateTask(ctx, req.Images, callback)
	if err != nil {
		return types.CreateTaskResponse{}, err
	}

	go s.processTask(context.Background(), taskID, taskImages)

	return types.CreateTaskResponse{
		TaskID: taskID,
		Status: "created",
	}, nil
}

func (s *TaskService) CreateUploadedTask(
	ctx context.Context,
	uploadedImages []types.UploadedTaskImage,
	callback *types.TaskCallbackRequest,
) (types.CreateTaskResponse, error) {
	if len(uploadedImages) == 0 {
		return types.CreateTaskResponse{}, fmt.Errorf("files list is empty")
	}

	uploadBatchID := uuid.NewString()
	uploadDir := filepath.Join(s.cfg.UploadDir, uploadBatchID)

	if err := os.MkdirAll(uploadDir, 0755); err != nil {
		return types.CreateTaskResponse{}, fmt.Errorf("create upload dir: %w", err)
	}

	taskImagesReq := make([]types.CreateTaskImageRequest, 0, len(uploadedImages))

	for i, uploaded := range uploadedImages {
		filename := sanitizeFilename(uploaded.Filename)

		if filename == "" {
			filename = fmt.Sprintf("image_%d.jpg", i+1)
		}

		if !hasSupportedImageExt(filename) {
			return types.CreateTaskResponse{}, fmt.Errorf("unsupported image extension: %s", filename)
		}

		externalID := strings.TrimSpace(uploaded.ID)
		if externalID == "" {
			externalID = filename
		}

		storedFilename := fmt.Sprintf("%02d_%s", i+1, filename)
		storedPath := filepath.Join(uploadDir, storedFilename)

		if err := os.WriteFile(storedPath, uploaded.Content, 0644); err != nil {
			return types.CreateTaskResponse{}, fmt.Errorf("save uploaded image: %w", err)
		}

		absPath, err := filepath.Abs(storedPath)
		if err != nil {
			return types.CreateTaskResponse{}, fmt.Errorf("resolve uploaded image path: %w", err)
		}

		taskImagesReq = append(taskImagesReq, types.CreateTaskImageRequest{
			ID:  externalID,
			URL: localFileURL(absPath),
		})
	}

	callback = normalizeCallback(callback)

	taskID, taskImages, err := s.repo.CreateTask(ctx, taskImagesReq, callback)
	if err != nil {
		return types.CreateTaskResponse{}, err
	}

	go s.processTask(context.Background(), taskID, taskImages)

	return types.CreateTaskResponse{
		TaskID: taskID,
		Status: "created",
	}, nil
}

func (s *TaskService) GetTaskDetails(
	ctx context.Context,
	taskID string,
) (types.TaskDetails, error) {
	return s.repo.GetTaskDetails(ctx, taskID)
}

func (s *TaskService) RenderAnnotatedImage(
	ctx context.Context,
	taskImageID string,
) ([]byte, string, error) {
	imageResult, err := s.repo.GetTaskImageResult(ctx, taskImageID)
	if err != nil {
		return nil, "", err
	}

	content, err := s.loadImage(ctx, imageResult.SourceURL)
	if err != nil {
		return nil, "", fmt.Errorf("load source image: %w", err)
	}

	src, _, err := image.Decode(bytes.NewReader(content))
	if err != nil {
		return nil, "", fmt.Errorf("decode image: %w", err)
	}

	bounds := src.Bounds()
	rgba := image.NewRGBA(bounds)

	imagedraw.Draw(rgba, bounds, src, bounds.Min, imagedraw.Src)

	for _, number := range imageResult.Numbers {
		drawBBox(rgba, number.BBoxXYXY, color.RGBA{R: 0, G: 255, B: 0, A: 255})
	}

	var out bytes.Buffer

	if err := jpeg.Encode(&out, rgba, &jpeg.Options{Quality: 92}); err != nil {
		return nil, "", fmt.Errorf("encode annotated image: %w", err)
	}

	return out.Bytes(), "image/jpeg", nil
}

func (s *TaskService) processTask(
	ctx context.Context,
	taskID string,
	images []types.TaskImage,
) {
	if err := s.repo.SetTaskStatus(ctx, taskID, "processing", nil); err != nil {
		return
	}

	for _, image := range images {
		if err := s.processImage(ctx, image); err != nil {
			msg := err.Error()
			_ = s.repo.SetTaskImageStatus(ctx, image.ID, "failed", &msg)
		}

		_ = s.repo.RefreshTaskProgress(ctx, taskID)
	}

	_ = s.repo.RefreshTaskProgress(ctx, taskID)
	s.sendCallbackIfNeeded(ctx, taskID)
}

func (s *TaskService) processImage(
	ctx context.Context,
	image types.TaskImage,
) error {
	if err := s.repo.SetTaskImageStatus(ctx, image.ID, "processing", nil); err != nil {
		return err
	}

	content, err := s.loadImage(ctx, image.SourceURL)
	if err != nil {
		return fmt.Errorf("load image: %w", err)
	}

	filename := filenameForML(image.ExternalID, image.SourceURL)

	mlResp, err := s.mlClient.RecognizeImageReader(
		ctx,
		filename,
		bytes.NewReader(content),
	)
	if err != nil {
		return fmt.Errorf("recognize image: %w", err)
	}

	numbers := convertMLRecognizedNumbers(mlResp.RecognizedNumbers)

	rawResponse, _ := json.Marshal(mlResp)

	if err := s.repo.SaveImageResults(ctx, image.ID, numbers, rawResponse); err != nil {
		return err
	}

	if err := s.repo.SetTaskImageStatus(ctx, image.ID, "completed", nil); err != nil {
		return err
	}

	return nil
}

func (s *TaskService) loadImage(
	ctx context.Context,
	imageURL string,
) ([]byte, error) {
	parsed, err := url.Parse(imageURL)
	if err == nil && parsed.Scheme == "file" {
		localPath, err := fileURLToPath(imageURL)
		if err != nil {
			return nil, err
		}

		content, err := os.ReadFile(localPath)
		if err != nil {
			return nil, fmt.Errorf("read local image file: %w", err)
		}

		if len(content) == 0 {
			return nil, fmt.Errorf("empty local image file")
		}

		return content, nil
	}

	return s.downloadImage(ctx, imageURL)
}

func (s *TaskService) downloadImage(
	ctx context.Context,
	imageURL string,
) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, imageURL, nil)
	if err != nil {
		return nil, fmt.Errorf("create download request: %w", err)
	}

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send download request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("image server status %d", resp.StatusCode)
	}

	content, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read image body: %w", err)
	}

	if len(content) == 0 {
		return nil, fmt.Errorf("empty image body")
	}

	return content, nil
}

func (s *TaskService) sendCallbackIfNeeded(ctx context.Context, taskID string) {
	details, err := s.repo.GetTaskDetails(ctx, taskID)
	if err != nil {
		return
	}

	if details.Callback == nil || !details.Callback.Enabled || details.Callback.URL == "" {
		return
	}

	if details.Status != "completed" &&
		details.Status != "completed_with_errors" &&
		details.Status != "failed" {
		return
	}

	var payload any

	if details.Callback.Mode == "status" {
		payload = map[string]any{
			"task_id":          details.ID,
			"status":           details.Status,
			"ready":            true,
			"total_images":     details.TotalImages,
			"processed_images": details.ProcessedImages,
		}
	} else {
		payload = details
	}

	body, err := json.Marshal(payload)
	if err != nil {
		msg := err.Error()
		_ = s.repo.SetCallbackResult(ctx, taskID, "failed", &msg)
		return
	}

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		details.Callback.URL,
		bytes.NewReader(body),
	)
	if err != nil {
		msg := err.Error()
		_ = s.repo.SetCallbackResult(ctx, taskID, "failed", &msg)
		return
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		msg := err.Error()
		_ = s.repo.SetCallbackResult(ctx, taskID, "failed", &msg)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg := fmt.Sprintf("callback returned status %d", resp.StatusCode)
		_ = s.repo.SetCallbackResult(ctx, taskID, "failed", &msg)
		return
	}

	_ = s.repo.SetCallbackResult(ctx, taskID, "sent", nil)
}

func normalizeCallback(callback *types.TaskCallbackRequest) *types.TaskCallbackRequest {
	if callback == nil {
		return nil
	}

	callback.URL = strings.TrimSpace(callback.URL)
	callback.Mode = strings.TrimSpace(callback.Mode)

	if callback.URL == "" {
		return nil
	}

	if callback.Mode == "" {
		callback.Mode = "result"
	}

	switch callback.Mode {
	case "result", "status":
		return callback
	default:
		callback.Mode = "result"
		return callback
	}
}

func filenameForML(externalID string, sourceURL string) string {
	if hasSupportedImageExt(externalID) {
		return externalID
	}

	parsedURL, err := url.Parse(sourceURL)
	if err == nil {
		base := filepath.Base(parsedURL.Path)
		if hasSupportedImageExt(base) {
			return base
		}
	}

	return "image.jpg"
}

func hasSupportedImageExt(filename string) bool {
	ext := strings.ToLower(filepath.Ext(filename))

	switch ext {
	case ".jpg", ".jpeg", ".png", ".bmp", ".webp":
		return true
	default:
		return false
	}
}

func sanitizeFilename(filename string) string {
	filename = strings.TrimSpace(filename)
	filename = filepath.Base(filename)
	filename = strings.ReplaceAll(filename, "\\", "_")
	filename = strings.ReplaceAll(filename, "/", "_")

	return filename
}

func localFileURL(path string) string {
	slashPath := filepath.ToSlash(path)

	if runtime.GOOS == "windows" && !strings.HasPrefix(slashPath, "/") {
		slashPath = "/" + slashPath
	}

	u := url.URL{
		Scheme: "file",
		Path:   slashPath,
	}

	return u.String()
}

func fileURLToPath(rawURL string) (string, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return "", fmt.Errorf("parse file url: %w", err)
	}

	if parsed.Scheme != "file" {
		return "", fmt.Errorf("not a file url: %s", rawURL)
	}

	path := parsed.Path

	if runtime.GOOS == "windows" && len(path) >= 3 && path[0] == '/' && path[2] == ':' {
		path = path[1:]
	}

	return filepath.FromSlash(path), nil
}

func convertMLRecognizedNumbers(items []ml.RecognizedNumber) []types.NumberResult {
	results := make([]types.NumberResult, 0, len(items))

	for _, item := range items {
		if item.Number == "" {
			continue
		}

		results = append(results, types.NumberResult{
			Number:             item.Number,
			AvgDigitConfidence: item.AvgDigitConfidence,
			DetectConfidence:   item.DetectConfidence,
			SegmentConfidence:  item.SegmentConfidence,
			CandidateScore:     item.CandidateScore,
			BBoxXYXY:           item.BBoxXYXY,
		})
	}

	return results
}

func drawBBox(img *image.RGBA, bbox []float64, c color.RGBA) {
	if len(bbox) != 4 {
		return
	}

	bounds := img.Bounds()

	x1 := clampInt(int(math.Round(bbox[0])), bounds.Min.X, bounds.Max.X-1)
	y1 := clampInt(int(math.Round(bbox[1])), bounds.Min.Y, bounds.Max.Y-1)
	x2 := clampInt(int(math.Round(bbox[2])), bounds.Min.X, bounds.Max.X-1)
	y2 := clampInt(int(math.Round(bbox[3])), bounds.Min.Y, bounds.Max.Y-1)

	if x2 <= x1 || y2 <= y1 {
		return
	}

	thickness := maxInt(2, minInt(bounds.Dx(), bounds.Dy())/250)

	for t := 0; t < thickness; t++ {
		drawRect(img, x1-t, y1-t, x2+t, y2+t, c)
	}
}

func drawRect(img *image.RGBA, x1 int, y1 int, x2 int, y2 int, c color.RGBA) {
	bounds := img.Bounds()

	x1 = clampInt(x1, bounds.Min.X, bounds.Max.X-1)
	y1 = clampInt(y1, bounds.Min.Y, bounds.Max.Y-1)
	x2 = clampInt(x2, bounds.Min.X, bounds.Max.X-1)
	y2 = clampInt(y2, bounds.Min.Y, bounds.Max.Y-1)

	for x := x1; x <= x2; x++ {
		img.SetRGBA(x, y1, c)
		img.SetRGBA(x, y2, c)
	}

	for y := y1; y <= y2; y++ {
		img.SetRGBA(x1, y, c)
		img.SetRGBA(x2, y, c)
	}
}

func clampInt(value int, minValue int, maxValue int) int {
	if value < minValue {
		return minValue
	}

	if value > maxValue {
		return maxValue
	}

	return value
}

func minInt(a int, b int) int {
	if a < b {
		return a
	}

	return b
}

func maxInt(a int, b int) int {
	if a > b {
		return a
	}

	return b
}
