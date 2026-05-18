package api

import (
	"errors"
	"fmt"
	"io"
	"net/http"

	"yolo12-number-system/backend/internal/repo"
	"yolo12-number-system/backend/internal/service"
	"yolo12-number-system/backend/internal/types"

	"github.com/gin-gonic/gin"
)

type Handler struct {
	taskService *service.TaskService
}

func NewHandler(taskService *service.TaskService) *Handler {
	return &Handler{
		taskService: taskService,
	}
}

func (h *Handler) Status(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
	})
}

func (h *Handler) CreateTask(c *gin.Context) {
	var req types.CreateTaskRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid json body",
		})
		return
	}

	resp, err := h.taskService.CreateTask(c.Request.Context(), req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": err.Error(),
		})
		return
	}

	c.JSON(http.StatusAccepted, resp)
}

func (h *Handler) UploadTask(c *gin.Context) {
	form, err := c.MultipartForm()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid multipart form",
		})
		return
	}

	fileHeaders := form.File["files"]

	if len(fileHeaders) == 0 {
		fileHeaders = form.File["file"]
	}

	if len(fileHeaders) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "field 'files' is required",
		})
		return
	}

	ids := form.Value["ids"]
	callback := parseCallbackFromMultipart(form.Value)

	uploadedImages := make([]types.UploadedTaskImage, 0, len(fileHeaders))

	for i, fileHeader := range fileHeaders {
		file, err := fileHeader.Open()
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": fmt.Sprintf("open uploaded file: %v", err),
			})
			return
		}

		content, err := io.ReadAll(file)
		_ = file.Close()

		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": fmt.Sprintf("read uploaded file: %v", err),
			})
			return
		}

		externalID := fileHeader.Filename
		if i < len(ids) && ids[i] != "" {
			externalID = ids[i]
		}

		uploadedImages = append(uploadedImages, types.UploadedTaskImage{
			ID:       externalID,
			Filename: fileHeader.Filename,
			Content:  content,
		})
	}

	resp, err := h.taskService.CreateUploadedTask(c.Request.Context(), uploadedImages, callback)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": err.Error(),
		})
		return
	}

	c.JSON(http.StatusAccepted, resp)
}

func (h *Handler) GetTask(c *gin.Context) {
	taskID := c.Param("id")

	if taskID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "task id is required",
		})
		return
	}

	details, err := h.taskService.GetTaskDetails(c.Request.Context(), taskID)
	if err != nil {
		if errors.Is(err, repo.ErrNotFound) {
			c.JSON(http.StatusNotFound, gin.H{
				"error": "task not found",
			})
			return
		}

		c.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}

	for i := range details.Images {
		details.Images[i].AnnotatedURL = fmt.Sprintf(
			"/api/task-images/%s/annotated",
			details.Images[i].ID,
		)
	}

	c.JSON(http.StatusOK, details)
}

func (h *Handler) GetAnnotatedTaskImage(c *gin.Context) {
	imageID := c.Param("id")

	if imageID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "task image id is required",
		})
		return
	}

	content, contentType, err := h.taskService.RenderAnnotatedImage(
		c.Request.Context(),
		imageID,
	)
	if err != nil {
		if errors.Is(err, repo.ErrNotFound) {
			c.JSON(http.StatusNotFound, gin.H{
				"error": "task image not found",
			})
			return
		}

		c.JSON(http.StatusInternalServerError, gin.H{
			"error": err.Error(),
		})
		return
	}

	c.Data(http.StatusOK, contentType, content)
}

func parseCallbackFromMultipart(values map[string][]string) *types.TaskCallbackRequest {
	callbackURL := firstFormValue(values, "callback_url")
	if callbackURL == "" {
		return nil
	}

	callbackMode := firstFormValue(values, "callback_mode")
	if callbackMode == "" {
		callbackMode = "result"
	}

	return &types.TaskCallbackRequest{
		URL:  callbackURL,
		Mode: callbackMode,
	}
}

func firstFormValue(values map[string][]string, key string) string {
	items := values[key]
	if len(items) == 0 {
		return ""
	}

	return items[0]
}
