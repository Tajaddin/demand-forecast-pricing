variable "name" {
  type    = string
  default = "dfp-api"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "mlflow_tracking_uri" {
  description = "MLflow server the API loads the production model from (with credentials in the URL if basic auth is on)"
  type        = string
  sensitive   = true
}

variable "api_keys" {
  description = "Comma-separated API keys for /forecast and /price"
  type        = string
  sensitive   = true
}

variable "rate_limit" {
  type    = string
  default = "60/minute"
}
