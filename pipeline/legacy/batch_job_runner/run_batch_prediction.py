import logging

from embedding_workflow.util import WorkflowConfig, run_workflow


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    results = run_workflow(WorkflowConfig.from_environment())
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
