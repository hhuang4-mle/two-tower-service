import os
import argparse
import logging
from kfp import compiler
from google.cloud import aiplatform

# Import the pipeline function
from pipeline import embedding_pipeline

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def _get_or_upload_model(
    project_id: str,
    location: str,
    model_resource_name: str,
    model_display_name: str,
    model_artifact_uri: str,
    serving_container_image_uri: str,
    truncate_dim: str
) -> str:
    """Ensure the model exists in Vertex AI and return its resource name."""
    aiplatform.init(project=project_id, location=location)

    def _model_resource_name_for_id(model_id: str) -> str:
        return f"projects/{project_id}/locations/{location}/models/{model_id}"

    if model_resource_name:
        model_id = model_resource_name.rsplit("/", maxsplit=1)[-1]
        final_resource_name = _model_resource_name_for_id(model_id)
        LOGGER.info(f"Using configured Vertex AI model {final_resource_name}")
        return final_resource_name

    # Check if model exists
    escaped_display_name = model_display_name.replace("\\", "\\\\").replace('"', '\\"')
    existing_models = aiplatform.Model.list(
        filter=f'display_name="{escaped_display_name}"',
        order_by="create_time desc",
    )
    existing_models = [m for m in existing_models if m.display_name == model_display_name]
    
    if existing_models:
        model = existing_models[0]
        model_id = model.resource_name.rsplit("/", maxsplit=1)[-1]
        final_resource_name = _model_resource_name_for_id(model_id)
        LOGGER.info(f"Reusing Vertex AI model {final_resource_name}")
        return final_resource_name

    LOGGER.info(f"No Vertex AI model named {model_display_name!r} found; uploading model")
    environment_variables = {"MODEL_ID": "/tmp/model"}
    if truncate_dim:
        environment_variables["TRUNCATE_DIM"] = truncate_dim

    uploaded_model = aiplatform.Model.upload(
        display_name=model_display_name,
        serving_container_image_uri=serving_container_image_uri,
        artifact_uri=model_artifact_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        serving_container_environment_variables=environment_variables,
        sync=True,
    )
    model_id = uploaded_model.resource_name.rsplit("/", maxsplit=1)[-1]
    final_resource_name = _model_resource_name_for_id(model_id)
    LOGGER.info(f"Uploaded Vertex AI model {final_resource_name}")
    return final_resource_name


def compile_and_submit(project_id, location, compile_only=False):
    pipeline_filename = "embedding_pipeline.yaml"
    
    # 1. Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=embedding_pipeline,
        package_path=pipeline_filename
    )
    LOGGER.info(f"Pipeline compiled successfully to {pipeline_filename}")
    
    if compile_only:
        return
        
    # 2. Prepare parameters
    dataset = f"{project_id}.{os.getenv('DATASET', 'two_tower')}"
    
    source_table = os.getenv("SOURCE_TABLE", f"{dataset}.super_product_text_final_17082026")
    text_column = os.getenv("TEXT_COLUMN", "product_embedding_text")
    input_table = os.getenv("INPUT_TABLE", f"{dataset}.test_input_embeddings")
    result_table = os.getenv("RESULT_TABLE", f"{dataset}.product_embeddings")
    
    # Read SQL templates
    with open("sql/preprocess.sql", "r") as f:
        preprocess_sql_template = f.read()
    with open("sql/postprocess.sql", "r") as f:
        postprocess_sql_template = f.read()
        
    # Format the preprocess query right now (since it uses static inputs)
    preprocess_query = preprocess_sql_template.format(
        input_table=input_table,
        source_table=source_table,
        text_column=text_column
    )

    # 3. Resolve the model outside the pipeline
    final_model_resource_name = _get_or_upload_model(
        project_id=project_id,
        location=location,
        model_resource_name=os.getenv("MODEL_RESOURCE_NAME", ""),
        model_display_name=os.getenv("MODEL_DISPLAY_NAME", "embeddinggemma-300m-bf16"),
        model_artifact_uri=os.getenv("MODEL_ARTIFACT_URI", "gs://wx-e1710759-2e30-40b2-a876-11588a57cb50/text-embedding-models/embeddinggemma-300m"),
        serving_container_image_uri=os.getenv("SERVING_CONTAINER_IMAGE_URI", "gcr.io/gcp-wow-rwds-ai-super-plo-dev/embeddinggemma-300m-bf16"),
        truncate_dim=os.getenv("TRUNCATE_DIM", "")
    )
    
    # 4. Submit to Vertex AI
    pipeline_params = {
        "project_id": project_id,
        "location": location,
        "model_resource_name": final_model_resource_name,
        "preprocess_query": preprocess_query,
        "postprocess_sql_template": postprocess_sql_template,
        "source_table": source_table,
        "text_column": text_column,
        "input_table": input_table,
        "output_dataset": os.getenv("OUTPUT_DATASET", dataset),
        "result_table": result_table,
        "display_name": os.getenv("BATCH_PREDICTION_DISPLAY_NAME", "embeddinggemma-300m-batch-prediction"),
        "cleanup_input_table": os.getenv("CLEANUP_INPUT_TABLE", "").strip().lower() in {"1", "true", "yes"}
    }
    
    aiplatform.init(project=project_id, location=location)
    pipeline_root = os.getenv("PIPELINE_ROOT") or (
        "gs://wx-1b759ae5-bd00-4de8-bf4d-427baa30e931/fs/vertex-ai-pipeline"
    )
    job = aiplatform.PipelineJob(
        display_name="embedding-workflow-pipeline",
        template_path=pipeline_filename,
        pipeline_root=pipeline_root,
        parameter_values=pipeline_params,
        enable_caching=False
    )
    
    LOGGER.info("Submitting pipeline job to Vertex AI...")
    job.submit()
    dashboard_uri = (
        f"https://console.cloud.google.com/vertex-ai/locations/{location}"
        f"/pipelines/runs/{job.name}?project={project_id}"
    )
    LOGGER.info(f"Pipeline submitted successfully. View it at: {dashboard_uri}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", default=os.getenv("PROJECT_ID", "gcp-wow-rwds-ai-super-plo-dev"))
    parser.add_argument("--location", default=os.getenv("LOCATION", os.getenv("REGION", "us-central1")))
    parser.add_argument("--compile-only", action="store_true", help="Only compile to YAML, do not submit")
    
    args = parser.parse_args()
    compile_and_submit(args.project_id, args.location, args.compile_only)
