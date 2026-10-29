from kfp import dsl
from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google_cloud_pipeline_components.types.artifact_types import VertexModel

@dsl.component(base_image="python:3.12", packages_to_install=["google-cloud-aiplatform==1.93.0"])
def build_postprocess_query(
    project_id: str,
    location: str,
    postprocess_sql_template: str,
    source_table: str,
    text_column: str,
    input_table: str,
    result_table: str,
    batchpredictionjob: dsl.Input[dsl.Artifact] = None,
) -> str:
    """Format validated postprocessing SQL, including the empty-source path."""
    prediction_source = (
        "SELECT CAST(NULL AS STRING) AS instances, "
        "CAST(NULL AS STRING) AS prediction WHERE FALSE"
    )
    if batchpredictionjob is not None:
        from google.cloud import aiplatform

        job_name = batchpredictionjob.uri.split("/v1/")[-1]
        aiplatform.init(project=project_id, location=location)
        job = aiplatform.BatchPredictionJob(job_name)
        output_info = job.output_info
        dataset = (output_info.bigquery_output_dataset or "").removeprefix("bq://")
        table = output_info.bigquery_output_table or ""
        if not dataset or not table:
            raise RuntimeError(
                "Vertex batch prediction succeeded without reporting a "
                f"BigQuery output table: {output_info}"
            )
        prediction_source = (
            f"SELECT instances, prediction FROM `{dataset}.{table}`"
        )

    query = postprocess_sql_template.format(
        prediction_source=prediction_source,
        source_table=source_table,
        text_column=text_column,
        input_table=input_table,
        result_table=result_table
    )
    return query


@dsl.component(base_image="python:3.12", packages_to_install=["google-cloud-bigquery==3.31.0"])
def count_input_rows(project_id: str, input_table: str) -> int:
    from google.cloud import bigquery

    return bigquery.Client(project=project_id).get_table(input_table).num_rows


@dsl.pipeline(
    name="embedding-batch-prediction-pipeline",
    description="BigQuery -> Vertex AI -> BigQuery embedding workflow using native components"
)
def embedding_pipeline(
    project_id: str,
    location: str,
    model_resource_name: str,
    preprocess_query: str,
    postprocess_sql_template: str,
    source_table: str,
    text_column: str,
    input_table: str,
    output_dataset: str,
    result_table: str,
    display_name: str = "embedding-batch-prediction",
    cleanup_input_table: bool = True,
):
    # Import the Vertex AI Model as an artifact
    model_artifact = dsl.importer(
        artifact_uri=f"https://{location}-aiplatform.googleapis.com/v1/{model_resource_name}",
        artifact_class=VertexModel,
        metadata={"resourceName": model_resource_name}
    )

    # Step 1: Preprocess (Native BigQuery Component)
    preprocess_task = BigqueryQueryJobOp(
        project=project_id,
        location=location,
        query=preprocess_query
    )
    preprocess_task.set_caching_options(False)

    input_count = count_input_rows(project_id=project_id, input_table=input_table)
    input_count.after(preprocess_task)
    input_count.set_caching_options(False)

    with dsl.If(input_count.output > 0, name="has-text-inputs"):
        batch_predict_task = ModelBatchPredictOp(
            project=project_id,
            location=location,
            model=model_artifact.output,
            job_display_name=display_name,
            bigquery_source_input_uri=f"bq://{input_table}",
            instances_format="bigquery",
            bigquery_destination_output_uri=f"bq://{output_dataset}",
            predictions_format="bigquery",
            machine_type="g2-standard-4",
            accelerator_type="NVIDIA_L4",
            accelerator_count=1,
            starting_replica_count=2,
            max_replica_count=2,
        )
        batch_predict_task.set_caching_options(False)
        format_query_task = build_postprocess_query(
            project_id=project_id,
            location=location,
            batchpredictionjob=batch_predict_task.outputs["batchpredictionjob"],
            postprocess_sql_template=postprocess_sql_template,
            source_table=source_table,
            text_column=text_column,
            input_table=input_table,
            result_table=result_table,
        )
        format_query_task.set_caching_options(False)
    with dsl.Else(name="empty-source"):
        empty_query_task = build_postprocess_query(
            project_id=project_id,
            location=location,
            postprocess_sql_template=postprocess_sql_template,
            source_table=source_table,
            text_column=text_column,
            input_table=input_table,
            result_table=result_table,
        )
        empty_query_task.set_caching_options(False)

    # Step 3: Postprocess (Native BigQuery Component)
    postprocess_task = BigqueryQueryJobOp(
        project=project_id,
        location=location,
        query=dsl.OneOf(format_query_task.output, empty_query_task.output)
    )
    postprocess_task.set_caching_options(False)

    # Step 4 (Optional): Cleanup (Native BigQuery Component)
    with dsl.If(cleanup_input_table == True, name="cleanup-condition"):
        cleanup_task = BigqueryQueryJobOp(
            project=project_id,
            location=location,
            query=f"DROP TABLE IF EXISTS `{input_table}`"
        )
        cleanup_task.after(postprocess_task)
        cleanup_task.set_caching_options(False)
