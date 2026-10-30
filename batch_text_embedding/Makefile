PROJECT_ID ?= gcp-wow-rwds-ai-super-plo-dev
IMAGE_NAME ?= embeddinggemma-300m-bf16
IMAGE_TAG ?= latest
IMAGE_URI := gcr.io/$(PROJECT_ID)/$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: build push

build:
	docker build --platform linux/amd64 --file Dockerfile --tag $(IMAGE_URI) .

push:
	gcloud builds submit --project=$(PROJECT_ID) --config=cloudbuild.yaml --substitutions=_IMAGE_URI=$(IMAGE_URI) .
