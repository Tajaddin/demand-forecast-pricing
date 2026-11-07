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
  description = "MLflow server the API loads the production model from, without credentials"
  type        = string
  sensitive   = true
}

variable "mlflow_tracking_username" {
  description = "MLflow basic-auth user; empty when the server has no auth"
  type        = string
  default     = ""
}

variable "mlflow_tracking_password" {
  type      = string
  default   = ""
  sensitive = true
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
