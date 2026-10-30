PYTHON ?= python
PROJECT_ID ?= gcp-wow-rwds-ai-super-plo-dev
VERTEX_LOCATION ?= us-central1
SERVICE_ACCOUNT ?= super-plo-dev@gcp-wow-rwds-ai-super-plo-dev.iam.gserviceaccount.com
REQUEST_BATCH_SIZE ?= 32
REPLICA_COUNT ?= 4
RETRY_ROUNDS ?= 1
CLEANUP_INTERMEDIATES ?= false

.PHONY: install compile submit

install:
	$(PYTHON) -m pip install -r requirements.txt

compile:
	$(PYTHON) submit_pipeline.py --compile-only

submit:
	@test -n "$(ENTITY_TYPE)" -a -n "$(SOURCE_TABLE)" -a -n "$(TEXT_COLUMNS)" \
		-a -n "$(MODEL_RESOURCE_NAME)" -a -n "$(BIGQUERY_LOCATION)" \
		-a -n "$(PIPELINE_ROOT)" -a -n "$(SERVICE_ACCOUNT)" || \
		{ echo "Set ENTITY_TYPE, SOURCE_TABLE, TEXT_COLUMNS, MODEL_RESOURCE_NAME, BIGQUERY_LOCATION, PIPELINE_ROOT and SERVICE_ACCOUNT"; exit 1; }
	$(PYTHON) submit_pipeline.py \
		--project-id "$(PROJECT_ID)" --vertex-location "$(VERTEX_LOCATION)" \
		--bigquery-location "$(BIGQUERY_LOCATION)" \
		--entity-type "$(ENTITY_TYPE)" --source-table "$(SOURCE_TABLE)" \
		--text-columns $(TEXT_COLUMNS) \
		--embedding-model-resource-name "$(MODEL_RESOURCE_NAME)" \
		--pipeline-root "$(PIPELINE_ROOT)" --service-account "$(SERVICE_ACCOUNT)" \
		--request-batch-size "$(REQUEST_BATCH_SIZE)" --replica-count "$(REPLICA_COUNT)" \
		--retry-rounds "$(RETRY_ROUNDS)" \
		$(if $(filter true,$(CLEANUP_INTERMEDIATES)),--cleanup-intermediates)
