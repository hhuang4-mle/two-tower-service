PROJECT_ID ?= gcp-wow-rwds-ai-super-plo-dev
REGION ?= us
VERTEX_REGION ?= us-central1
CLOUD_RUN_REGION ?= us-central1
CLOUD_RUN_JOB_NAME ?= embeddinggemma-batch-job-runner
CLOUD_RUN_TASK_TIMEOUT ?= 20h
IMAGE_NAME ?= text-embedding-batch-prediction
IMAGE_TAG ?= latest
IMAGE_URI := gcr.io/$(PROJECT_ID)/$(IMAGE_NAME):$(IMAGE_TAG)
PYTHON ?= python3
DOCKER_CONTEXT := .
PREDICTOR_DOCKERFILE := predictor/Dockerfile
BATCH_JOB_RUNNER_DOCKERFILE := batch_job_runner/Dockerfile
PREDICTOR_CLOUDBUILD := predictor/cloudbuild.yaml
BATCH_JOB_RUNNER_CLOUDBUILD := batch_job_runner/cloudbuild.yaml
empty :=
space := $(empty) $(empty)

MODEL_RESOURCE_NAME ?=
TRUNCATE_DIM ?= 768

# Manually specify the service account to use for Cloud Run job execution. 
# If not specified, the default Compute Engine service account will be used.
DEFAULT_RUNTIME_SA ?= super-plo-dev@gcp-wow-rwds-ai-super-plo-dev.iam.gserviceaccount.com

.PHONY: build push build-batch-job-runner push-batch-job-runner submit run \
	run-customer-test run-customer-test-cloud-run

build:
	docker build --platform linux/amd64 --file $(PREDICTOR_DOCKERFILE) --tag $(IMAGE_URI) $(DOCKER_CONTEXT)

push:
	gcloud builds submit --config=$(PREDICTOR_CLOUDBUILD) --substitutions=_IMAGE_URI=$(IMAGE_URI) $(DOCKER_CONTEXT)

BATCH_JOB_RUNNER_IMAGE_NAME ?= embeddinggemma-batch-job-runner
BATCH_JOB_RUNNER_IMAGE_URI := gcr.io/$(PROJECT_ID)/$(BATCH_JOB_RUNNER_IMAGE_NAME):$(IMAGE_TAG)

# Customer test configuration
CUSTOMER_TEST_SOURCE_TABLE ?= gcp-wow-rwds-ai-super-plo-dev.two_tower.super_long_term_customer_text_small
CUSTOMER_TEST_TEXT_COLUMN ?= customer_embedding_text
CUSTOMER_TEST_INPUT_TABLE ?= gcp-wow-rwds-ai-super-plo-dev.two_tower.customer_embedding_input_test
CUSTOMER_TEST_RESULT_TABLE ?= gcp-wow-rwds-ai-super-plo-dev.two_tower.customer_embeddings_test
CUSTOMER_TEST_ENV_VARS := \
	PROJECT_ID=$(PROJECT_ID),\
	LOCATION=$(VERTEX_REGION),\
	SOURCE_TABLE=$(CUSTOMER_TEST_SOURCE_TABLE),\
	TEXT_COLUMN=$(CUSTOMER_TEST_TEXT_COLUMN),\
	INPUT_TABLE=$(CUSTOMER_TEST_INPUT_TABLE),\
	RESULT_TABLE=$(CUSTOMER_TEST_RESULT_TABLE),\
	SERVING_CONTAINER_IMAGE_URI=$(IMAGE_URI),\
	MODEL_RESOURCE_NAME=$(MODEL_RESOURCE_NAME),\
	TRUNCATE_DIM=$(TRUNCATE_DIM)

build-batch-job-runner:
	docker build --platform linux/amd64 --file $(BATCH_JOB_RUNNER_DOCKERFILE) --tag $(BATCH_JOB_RUNNER_IMAGE_URI) $(DOCKER_CONTEXT)

push-batch-job-runner:
	gcloud builds submit --config=$(BATCH_JOB_RUNNER_CLOUDBUILD) --substitutions=_IMAGE_URI=$(BATCH_JOB_RUNNER_IMAGE_URI) $(DOCKER_CONTEXT)

submit:
	SERVING_CONTAINER_IMAGE_URI=$(IMAGE_URI) \
	MODEL_RESOURCE_NAME=$(MODEL_RESOURCE_NAME) \
	TRUNCATE_DIM=$(TRUNCATE_DIM) \
	PROJECT_ID=$(PROJECT_ID) \
	REGION=$(VERTEX_REGION) \
	PYTHONPATH=shared $(PYTHON) batch_job_runner/run_batch_prediction.py

run:
	PROJECT_ID=$(PROJECT_ID) \
	LOCATION=$(VERTEX_REGION) \
	MODEL_RESOURCE_NAME=$(MODEL_RESOURCE_NAME) \
	PYTHONPATH=shared $(PYTHON) batch_job_runner/run_batch_prediction.py

run-customer-test:
	PROJECT_ID=$(PROJECT_ID) \
	LOCATION=$(VERTEX_REGION) \
	SOURCE_TABLE=$(CUSTOMER_TEST_SOURCE_TABLE) \
	TEXT_COLUMN=$(CUSTOMER_TEST_TEXT_COLUMN) \
	INPUT_TABLE=$(CUSTOMER_TEST_INPUT_TABLE) \
	RESULT_TABLE=$(CUSTOMER_TEST_RESULT_TABLE) \
	SERVING_CONTAINER_IMAGE_URI=$(IMAGE_URI) \
	MODEL_RESOURCE_NAME=$(MODEL_RESOURCE_NAME) \
	TRUNCATE_DIM=$(TRUNCATE_DIM) \
	PYTHONPATH=shared $(PYTHON) batch_job_runner/run_batch_prediction.py

run-customer-test-cloud-run: 
	gcloud run jobs deploy $(CLOUD_RUN_JOB_NAME) \
		--image=$(BATCH_JOB_RUNNER_IMAGE_URI) \
		--project=$(PROJECT_ID) \
		--region=$(CLOUD_RUN_REGION) \
		--task-timeout=$(CLOUD_RUN_TASK_TIMEOUT) \
		--max-retries=0 \
		--service-account=$(DEFAULT_RUNTIME_SA) \
		--set-env-vars=$(subst $(space),,$(CUSTOMER_TEST_ENV_VARS))
	gcloud run jobs execute $(CLOUD_RUN_JOB_NAME) \
		--project=$(PROJECT_ID) \
		--region=$(CLOUD_RUN_REGION) 
