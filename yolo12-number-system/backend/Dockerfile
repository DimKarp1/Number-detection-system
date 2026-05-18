FROM golang:1.25-alpine AS builder

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /out/backend ./cmd/server


FROM alpine:3.20

WORKDIR /app

COPY --from=builder /out/backend /app/backend
COPY migrations /app/migrations

RUN mkdir -p /app/uploads

EXPOSE 8080

CMD ["/app/backend"]
