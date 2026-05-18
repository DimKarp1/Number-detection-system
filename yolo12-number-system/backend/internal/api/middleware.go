package api

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func APIKeyMiddleware(apiKey string) gin.HandlerFunc {
	return func(c *gin.Context) {
		provided := c.GetHeader("X-API-Key")

		if provided == "" || provided != apiKey {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "invalid or missing X-API-Key",
			})
			c.Abort()
			return
		}

		c.Next()
	}
}
