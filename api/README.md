# Recommendation API (placeholder)

This directory reserves the future API-only application, which will have its
own Docker image, separate from the text-embedding and batch-projection images.
No API code, Dockerfile, or deployment is implemented yet.

The intended online flow is to encode in-session customer signals, combine
them with precomputed customer embeddings, and perform product recommendation.
The encoding model, combination logic, endpoints, embedding access, product
retrieval/ranking, and hosting will be designed later. Do not assume the API
will be model-free.

Batch orchestration belongs to Vertex AI Pipelines, not this API. See
[`ARCHITECTURE.md`](../ARCHITECTURE.md) for the batch design and implementation
plan.
