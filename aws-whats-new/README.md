# AWS What's New Tracker

Private directory for tracking AWS service announcements. Not published to GitHub Pages.

## Structure

```
aws-whats-new/
├── README.md              # This file
├── YYYY-MM-DD.md          # Daily digests
├── YYYY-WXX.md            # Weekly summaries (optional)
└── by-service/            # Per-service tracking (optional)
    ├── bedrock.md
    ├── sagemaker.md
    └── ...
```

## Source

- RSS feed: https://aws.amazon.com/about-aws/whats-new/recent/feed/
- Web page: https://aws.amazon.com/about-aws/whats-new/
- API: `aws whatsnew` (if available)

## Digest Format

Each daily/weekly digest follows this structure:

```markdown
# AWS What's New: YYYY-MM-DD

## Summary
- Total announcements: N
- Key themes: ...

## Announcements

### [Service Category]

#### Announcement Title
- **Date**: YYYY-MM-DD
- **Service**: Service Name
- **Region**: Global / us-east-1 / etc.
- **Link**: https://aws.amazon.com/about-aws/whats-new/...
- **Impact**: High / Medium / Low
- **TL;DR**: One-line summary
```

## Tags of Interest

Focus areas for tracking:

- **AI/ML**: Bedrock, SageMaker, Q Developer, Titan
- **Compute**: Lambda, ECS, EKS, EC2
- **Data**: S3, DynamoDB, Aurora, Redshift
- **Security**: IAM, GuardDuty, Security Hub
- **Developer Tools**: CodeBuild, CodePipeline, CloudFormation, CDK
- **Networking**: VPC, CloudFront, Route 53
