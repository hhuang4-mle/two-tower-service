import argparse
from pathlib import Path

from kfp import compiler

from pipeline import embedding_pipeline


def main():
    parser = argparse.ArgumentParser(description="Compile or submit the encoder-only Vertex pipeline")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--template", default=str(Path(__file__).with_name("embedding_pipeline.yaml")))
    parser.add_argument("--project-id", default="gcp-wow-rwds-ai-super-plo-dev")
    parser.add_argument("--vertex-location", default="us-central1")
    parser.add_argument("--bigquery-location")
    parser.add_argument("--entity-type", choices=["customer", "product"])
    parser.add_argument("--source-table")
    parser.add_argument("--text-columns", nargs="+")
    parser.add_argument("--embedding-model-resource-name")
    parser.add_argument("--pipeline-root")
    parser.add_argument(
        "--service-account",
        default="super-plo-dev@gcp-wow-rwds-ai-super-plo-dev.iam.gserviceaccount.com",
    )
    parser.add_argument("--request-batch-size", type=int, default=32)
    parser.add_argument("--replica-count", type=int, default=4)
    parser.add_argument("--retry-rounds", type=int, choices=range(3), default=2)
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()
    if not args.compile_only:
        for name in (
            "bigquery_location", "entity_type", "source_table", "text_columns",
            "embedding_model_resource_name", "pipeline_root", "service_account",
        ):
            if not getattr(args, name):
                parser.error(f"--{name.replace('_', '-')} is required for submission")
        expected = 2 if args.entity_type == "customer" else 1
        if len(args.text_columns) != expected:
            parser.error("Customer runs need two ordered text columns; product runs need one")
        if args.request_batch_size <= 0 or args.replica_count <= 0:
            parser.error("Batch size and replica count must be positive")
    compiler.Compiler().compile(
        pipeline_func=embedding_pipeline, package_path=args.template
    )
    print(f"Compiled {args.template}")
    if args.compile_only:
        return

    from google.cloud import aiplatform

    parameters = {
        name: getattr(args, name) for name in (
            "project_id", "vertex_location", "bigquery_location", "entity_type",
            "source_table", "text_columns", "embedding_model_resource_name",
            "request_batch_size", "replica_count", "retry_rounds", "cleanup_intermediates",
            "service_account",
        )
    }
    aiplatform.init(project=args.project_id, location=args.vertex_location)
    job = aiplatform.PipelineJob(
        display_name=f"embeddinggemma-{args.entity_type}",
        template_path=args.template,
        pipeline_root=args.pipeline_root,
        parameter_values=parameters,
        enable_caching=False,
    )
    job.submit(service_account=args.service_account)
    print(job.resource_name)


if __name__ == "__main__":
    main()
