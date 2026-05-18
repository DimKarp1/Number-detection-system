package api

import (
	"github.com/gin-gonic/gin"
)

func NewRouter(handler *Handler, apiKey string) *gin.Engine {
	router := gin.Default()

	router.MaxMultipartMemory = 256 << 20

	api := router.Group("/api")

	api.GET("/status", handler.Status)

	protected := api.Group("")
	protected.Use(APIKeyMiddleware(apiKey))

	protected.POST("/tasks", handler.CreateTask)
	protected.POST("/tasks/upload", handler.UploadTask)
	protected.GET("/tasks/:id", handler.GetTask)
	protected.GET("/task-images/:id/annotated", handler.GetAnnotatedTaskImage)

	return router
}
