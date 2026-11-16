# Deploying to AWS (App Runner)

The service is a stateless container, so App Runner is the simplest fit: no load balancer,
cluster, or VPC to manage, and it scales to zero traffic cheaply. `infra/terraform/main.tf`
creates:

* an ECR repository (scan-on-push enabled),
* a Secrets Manager secret for `ANTHROPIC_API_KEY` (the value is set by you, outside Terraform,
  so it never lands in state or git),
* IAM roles that let App Runner pull the image and read only that one secret,
* the App Runner service with a `/healthz` health check.

## Steps

```bash
cd infra/terraform
terraform init
terraform apply -target=aws_ecr_repository.analyst -target=aws_secretsmanager_secret.anthropic_key

# 1. Store the key (skip if you only want the offline mock provider)
aws secretsmanager put-secret-value \
  --secret-id multi-agent-llm-analyst/anthropic-api-key --secret-string "$ANTHROPIC_API_KEY"

# 2. Build and push the image
REPO=$(terraform output -raw ecr_repository_url)
aws ecr get-login-password | docker login --username AWS --password-stdin "${REPO%%/*}"
docker build -t "$REPO:latest" ../.. && docker push "$REPO:latest"

# 3. Create the service
terraform apply -var provider_name=anthropic   # or leave the default "mock"
curl -X POST "$(terraform output -raw service_url)/v1/analyze" \
  -H 'content-type: application/json' -d '{"question":"Where is Lumora Systems headquartered?"}'
```

## Production notes

* **Cost control.** Every request is capped by `Budget` (LLM calls and USD per request). Put
  an API Gateway usage plan or WAF rate rule in front for per-client limits.
* **Observability.** Traces are JSON span trees; ship them to CloudWatch Logs (stdout) or map
  them onto OpenTelemetry spans for X-Ray. Track `cost_usd`, `llm_calls`, and critic rejection
  rate per request as metrics.
* **Scaling the corpus.** BM25 in memory is fine for thousands of documents. Beyond that, swap
  `BM25Retriever` for OpenSearch or a vector store behind the same `search(query, k)` method.
* **Teardown.** `terraform destroy`.
