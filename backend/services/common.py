"""
Helpers shared by the classical and hybrid arms.

These are pure label/type heuristics with no detection logic in them, so both
arms can use them without the comparison becoming circular. The connection
DETECTOR is deliberately not shared — see services/connection_detector.py.

Definitions live here rather than in classical_pipeline.py so that the hybrid
arm never imports from the frozen control arm.
"""

from __future__ import annotations

import re

# ── Type classification keywords ──────────────────────────────────────────────
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "database":      ["database", "db", "postgres", "postgresql", "mysql", "mongo", "mongodb",
                      "dynamo", "dynamodb", "sql", "sqlite", "oracle", "rds", "cassandra",
                      "aurora", "redshift", "datastore", "bigtable", "spanner"],
    "cache":         ["cache", "redis", "memcache", "memcached", "elasticache"],
    "queue":         ["queue", "kafka", "rabbitmq", "rabbit", "sqs", "sns", "pubsub", "pub/sub",
                      "eventbus", "event bus", "message bus", "broker", "kinesis", "celery",
                      "eventbridge", "topic"],
    "gateway":       ["gateway", "api gw", "api gateway", "apigw", "kong", "envoy", "ingress",
                      "proxy", "reverse proxy", "nginx"],
    "load_balancer": ["load balancer", "load-balancer", "loadbalancer", "alb", "elb", "nlb",
                      "balancer", "haproxy", "traffic manager"],
    "cdn":           ["cdn", "cloudfront", "akamai", "fastly", "edge", "content delivery"],
    "storage":       ["s3", "blob", "storage", "gcs", "file store", "filestore", "object store",
                      "bucket", "efs", "ebs", "minio"],
    "client":        ["client", "browser", "mobile", "user", "frontend", "front-end", "web app",
                      "webapp", "ios", "android", "spa", "ui"],
    "monitoring":    ["monitoring", "monitor", "prometheus", "grafana", "datadog", "cloudwatch",
                      "logging", "tracing", "jaeger", "kibana", "elk", "sentry"],
    "notification":  ["notification", "notify", "email", "smtp", "ses", "twilio", "push",
                      "firebase messaging", "fcm", "webhook"],
}

_AWS_KEYWORDS = {"aws", "ec2", "s3", "lambda", "cloudfront", "rds", "sqs", "sns", "eks", "ecs",
                 "fargate", "dynamodb", "elasticache", "alb", "elb", "cloudwatch", "route53",
                 "kinesis", "aurora", "redshift", "eventbridge", "cognito", "amazon"}
_C4_KEYWORDS  = {"person", "system", "container", "component", "bounded", "context", "c4",
                 "boundary"}
_UML_KEYWORDS = {"actor", "interface", "class", "abstract", "package", "stereotype", "extends",
                 "implements", "usecase", "use case"}

_STOPWORDS = {"the", "and", "for", "with", "via", "to", "of", "a", "an", "or", "in", "on",
              "by", "is", "are"}

# Short tokens (<=3 alpha chars) that are still meaningful. Anything <=3 chars
# NOT listed here is treated as OCR garbage from icon glyphs.
_KNOWN_SHORT = {
    "elb", "alb", "nlb", "rds", "cdn", "ec2", "s3", "api", "vpc", "iam", "sns",
    "sqs", "ecs", "eks", "emr", "ses", "efs", "ebs", "az", "lb", "db", "ui", "ux",
    "app", "web", "dns", "ssl", "tls", "vm", "vms", "waf", "kms", "ec", "elk",
    "kafka", "id", "cli", "sdk", "gpu", "cpu", "iot", "mq", "fcm", "spa",
}

# Group boundaries, not components. Used to tag containers rather than drop them.
CONTAINER_KEYWORDS = {
    "vpc", "subnet", "availability zone", "availability-zone", "region",
    "private subnet", "public subnet", "security group", "resource group",
    "namespace", "cluster boundary", "system boundary", "boundary",
    "data center", "datacenter", "on-premises", "on premises", "network",
}


def classify_type(text: str) -> str:
    lower = f" {text.lower()} "
    for component_type, keywords in _TYPE_KEYWORDS.items():
        for kw in keywords:
            if f" {kw} " in lower or lower.strip().endswith(kw) or lower.strip().startswith(kw):
                return component_type
    return "service"


def infer_diagram_standard(names: list[str]) -> str:
    words = set(re.split(r"[\s,./_-]+", " ".join(names).lower()))
    if words & _AWS_KEYWORDS:
        return "aws"
    if words & _C4_KEYWORDS:
        return "c4"
    if words & _UML_KEYWORDS:
        return "uml"
    return "informal"


def infer_arch_type(component_types: list[str]) -> str:
    if "queue" in component_types:
        return "event_driven"
    if component_types.count("service") >= 3:
        return "microservices"
    return "other"


def complexity(n: int) -> str:
    if n < 8:
        return "low"
    if n <= 14:
        return "medium"
    return "high"


def is_noise(name: str) -> bool:
    """Reject OCR garbage: icon glyphs ('ee', 'oO'), symbol soup, bare stopwords."""
    letters = re.sub(r"[^A-Za-z]", "", name)
    if len(letters) < 2:
        return True
    if name.lower() in _STOPWORDS:
        return True

    # Low alpha ratio = symbol soup
    non_space = re.sub(r"\s", "", name)
    if non_space and len(letters) / len(non_space) < 0.5:
        return True

    # A real label needs one substantial token: >=4 alpha chars, or a known
    # short acronym. Kills "ee", "ol Lae", "Ey", "fal".
    for tok in name.split():
        tok_alpha = re.sub(r"[^A-Za-z]", "", tok)
        if len(tok_alpha) >= 4 or tok_alpha.lower() in _KNOWN_SHORT:
            return False
    return True


def looks_like_container(name: str) -> bool:
    lower = name.lower()
    return any(kw in lower for kw in CONTAINER_KEYWORDS)
