# GPT-6 Astra on Amazon Bedrock: configuration findings

Verified on 2026-09-09 using AWS official documentation, OpenAI official
documentation, and read-only AWS APIs against the currently configured account.

## Account and model status

- `openai.gpt-6-astra` is `ACTIVE` in `us-east-1` and `us-east-2`.
- The model accepts text and image input, returns text, supports streaming, and
  can only be invoked through an inference profile.
- Both profiles are active:
  - `global.openai.gpt-6-astra`
  - `us.openai.gpt-6-astra`
- The account reports:
  - agreement: `AVAILABLE`
  - authorization: `AUTHORIZED`
  - entitlement: `AVAILABLE`
  - region: `AVAILABLE`
- No Astra-specific quota currently appears in the Bedrock Service Quotas API.

## Routing recommendation

Use `global.openai.gpt-6-astra` by default. It is cheaper and permits global
cross-region routing. Use `us.openai.gpt-6-astra` only when processing must
remain in the US routing geography.

The existing Codex configuration uses `us-east-2`; Astra and both inference
profiles are active there, so no region change is required.

## Price verification

The Marketplace agreement rate card confirms the supplied standard prices:

| Route | Input | Cache read | 30m cache write | Output |
| --- | ---: | ---: | ---: | ---: |
| Global standard | 10 | 1 | 12.5 | 50 |
| US standard | 11 | 1.1 | 13.75 | 55 |
| Global long context | 20 | 2 | 25 | 75 |
| US long context | 22 | 2.2 | 27.5 | 82.5 |

All prices are USD per million tokens. The rate card also contains Flex prices
at roughly half standard and Priority prices at roughly twice standard. The
public pricing page does not yet provide the complete Astra matrix.

## Application API

For AWS SDK code, use Bedrock Runtime `Converse` or `ConverseStream` with the
inference profile ID:

```python
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-2")
response = client.converse(
    modelId="global.openai.gpt-6-astra",
    messages=[
        {"role": "user", "content": [{"text": "Reply with OK"}]},
    ],
    inferenceConfig={"maxTokens": 32},
)
print(response["output"]["message"]["content"][0]["text"])
```

For OpenAI-compatible clients, use:

```text
https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1
```

with `model="global.openai.gpt-6-astra"` and a Bedrock API key as the bearer
token. Do not place the API key directly in source code or Codex TOML.

## Codex

OpenAI's current Codex Bedrock page documents the built-in
`amazon-bedrock` provider against the Bedrock Mantle endpoint. Astra is not yet
listed in the documented Mantle model set. Therefore, changing only the model
name in the existing built-in provider is not a reliable configuration yet.

For immediate use, define a custom provider against the Bedrock Runtime
Responses endpoint:

```toml
model = "global.openai.gpt-6-astra"
model_provider = "bedrock-astra-runtime"
model_reasoning_effort = "high"

[model_providers.bedrock-astra-runtime]
name = "Amazon Bedrock Astra Runtime"
base_url = "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1"
env_key = "AWS_BEARER_TOKEN_BEDROCK"
wire_api = "responses"
```

The simplest authentication path is a long-term Bedrock API key stored in
`AWS_BEARER_TOKEN_BEDROCK`. To continue using the existing AWS Profile instead,
install AWS's Bedrock token generator and configure Codex command-backed
authentication; this adds a dependency and should be tested before replacing
the working GPT-5.6 Sol provider.

Keep the current context settings initially. Although the supplied launch
information says one million input tokens, the public Astra model card is not
yet available, so increasing Codex's context and auto-compaction limits should
wait for the official limit or a controlled test.

## IAM minimum

The calling principal needs:

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream"
  ],
  "Resource": [
    "arn:aws:bedrock:us-east-2:ACCOUNT_ID:inference-profile/global.openai.gpt-6-astra",
    "arn:aws:bedrock:*::foundation-model/openai.gpt-6-astra"
  ]
}
```

Add the US inference profile ARN instead if using US-only routing.

## Retention and tools

- The current AWS abuse-detection page lists earlier OpenAI GPT models but has
  not yet added Astra. Treat the launch statement about classifier-flagged
  traffic being retained for up to 30 days as provisional until this page is
  updated.
- Bedrock data retention can be controlled at account or project scope. ZDR is
  evaluated per account and per model through the AWS account team.
- Runtime OpenAI-compatible endpoints and Mantle are different surfaces. Since
  Astra is currently runtime-only, do not depend on Mantle-only projects,
  background mode, or hosted tools until Astra appears in Mantle's model list.

## Primary sources

- OpenAI, Amazon Bedrock integration:
  https://developers.openai.com/cookbook/articles/gpt-oss/run-codex-on-amazon-bedrock/
- OpenAI, Codex model providers:
  https://developers.openai.com/codex/models/model-providers/
- OpenAI, Codex configuration reference:
  https://developers.openai.com/codex/config-reference/
- AWS, OpenAI-compatible APIs:
  https://docs.aws.amazon.com/bedrock/latest/userguide/inference-api-openai.html
- AWS, Bedrock API keys:
  https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html
- AWS, data retention:
  https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html
- AWS, abuse detection:
  https://docs.aws.amazon.com/bedrock/latest/userguide/abuse-detection.html
