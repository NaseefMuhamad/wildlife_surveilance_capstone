"""Fine-tuned inference by default; explicitly pass --model for the COCO baseline."""
from pipeline_cli import predict_main

if __name__ == '__main__':
    predict_main()
