output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "service_url" {
  value = "https://${aws_apprunner_service.api.service_url}"
}
