package config

import (
	"errors"
	"os"
	"strconv"
)

type Config struct {
	HTTPAddr     string
	DatabaseURL  string
	APIKey       string
	MLServiceURL string

	TaskWorkers int
	UploadDir   string
}

func Load() (Config, error) {
	cfg := Config{
		HTTPAddr:     getEnv("HTTP_ADDR", ":8080"),
		DatabaseURL:  os.Getenv("DATABASE_URL"),
		APIKey:       os.Getenv("API_KEY"),
		MLServiceURL: getEnv("ML_SERVICE_URL", "http://127.0.0.1:8000"),

		TaskWorkers: getEnvInt("TASK_WORKERS", 2),
		UploadDir:   getEnv("UPLOAD_DIR", "uploads"),
	}

	if cfg.DatabaseURL == "" {
		return Config{}, errors.New("DATABASE_URL is required")
	}

	if cfg.APIKey == "" {
		return Config{}, errors.New("API_KEY is required")
	}

	return cfg, nil
}

func getEnv(key string, fallback string) string {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}

	return value
}

func getEnvInt(key string, fallback int) int {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}

	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}

	return parsed
}
