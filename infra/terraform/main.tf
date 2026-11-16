# Minimal AWS deployment: container image in ECR, served by App Runner, API key in Secrets
# Manager. See docs/deploy-aws.md for the full walkthrough.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "provider_name" {
  type        = string
  default     = "mock"
  description = "mock or anthropic"
}

resource "aws_ecr_repository" "analyst" {
  name                 = "multi-agent-llm-analyst"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

# The secret's value is set out-of-band (console or CLI), never in Terraform state.
resource "aws_secretsmanager_secret" "anthropic_key" {
  name = "multi-agent-llm-analyst/anthropic-api-key"
}

data "aws_iam_policy_document" "apprunner_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["build.apprunner.amazonaws.com", "tasks.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "access" {
  name               = "analyst-apprunner-ecr-access"
  assume_role_policy = data.aws_iam_policy_document.apprunner_assume.json
}

resource "aws_iam_role_policy_attachment" "ecr" {
  role       = aws_iam_role.access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

resource "aws_iam_role" "instance" {
  name               = "analyst-apprunner-instance"
  assume_role_policy = data.aws_iam_policy_document.apprunner_assume.json
}

resource "aws_iam_role_policy" "read_secret" {
  role = aws_iam_role.instance.id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = aws_secretsmanager_secret.anthropic_key.arn }]
  })
}

resource "aws_apprunner_service" "analyst" {
  service_name = "multi-agent-llm-analyst"

  source_configuration {
    authentication_configuration { access_role_arn = aws_iam_role.access.arn }
    auto_deployments_enabled = false
    image_repository {
      image_identifier      = "${aws_ecr_repository.analyst.repository_url}:${var.image_tag}"
      image_repository_type = "ECR"
      image_configuration {
        port                          = "8000"
        runtime_environment_variables = { ANALYST_PROVIDER = var.provider_name }
        runtime_environment_secrets   = { ANTHROPIC_API_KEY = aws_secretsmanager_secret.anthropic_key.arn }
      }
    }
  }

  instance_configuration {
    cpu               = "1024"
    memory            = "2048"
    instance_role_arn = aws_iam_role.instance.arn
  }

  health_check_configuration {
    protocol = "HTTP"
    path     = "/healthz"
  }
}

output "ecr_repository_url" { value = aws_ecr_repository.analyst.repository_url }
output "service_url" { value = "https://${aws_apprunner_service.analyst.service_url}" }
