// AWS path: the API image in ECR, served by App Runner. MLflow is passed in as a URL (for example the Fly
// MLflow app or a managed one); App Runner pulls the image and scales to zero-ish on the smallest size.
terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.69"
    }
  }
}

provider "aws" {
  region = var.region
}

resource "aws_ecr_repository" "api" {
  name                 = var.name
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "keep_last" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the last 10 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}

data "aws_iam_policy_document" "apprunner_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["build.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_access" {
  name               = "${var.name}-apprunner-access"
  assume_role_policy = data.aws_iam_policy_document.apprunner_assume.json
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  role       = aws_iam_role.apprunner_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

resource "aws_apprunner_service" "api" {
  service_name = var.name

  source_configuration {
    auto_deployments_enabled = true
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_access.arn
    }
    image_repository {
      image_identifier      = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
      image_repository_type = "ECR"
      image_configuration {
        port = "8000"
        runtime_environment_variables = {
          DFP_LOG_LEVEL   = "info"
          DFP_RATE_LIMIT  = var.rate_limit
          DFP_REPORTS_DIR = "/app/reports"
        }
        runtime_environment_secrets = {
          MLFLOW_TRACKING_URI = aws_ssm_parameter.mlflow_uri.arn
          DFP_API_KEYS        = aws_ssm_parameter.api_keys.arn
        }
      }
    }
  }

  instance_configuration {
    cpu               = "256"
    memory            = "512"
    instance_role_arn = aws_iam_role.apprunner_instance.arn
  }

  health_check_configuration {
    protocol = "HTTP"
    path     = "/metrics"
    interval = 10
    timeout  = 5
  }

  depends_on = [aws_iam_role_policy_attachment.apprunner_ecr]
}

resource "aws_ssm_parameter" "mlflow_uri" {
  name  = "/${var.name}/mlflow-tracking-uri"
  type  = "SecureString"
  value = var.mlflow_tracking_uri
}

resource "aws_ssm_parameter" "api_keys" {
  name  = "/${var.name}/api-keys"
  type  = "SecureString"
  value = var.api_keys
}

data "aws_iam_policy_document" "instance_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["tasks.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_instance" {
  name               = "${var.name}-apprunner-instance"
  assume_role_policy = data.aws_iam_policy_document.instance_assume.json
}

data "aws_iam_policy_document" "read_params" {
  statement {
    actions   = ["ssm:GetParameters", "ssm:GetParameter"]
    resources = [aws_ssm_parameter.mlflow_uri.arn, aws_ssm_parameter.api_keys.arn]
  }
}

resource "aws_iam_role_policy" "read_params" {
  role   = aws_iam_role.apprunner_instance.id
  policy = data.aws_iam_policy_document.read_params.json
}
