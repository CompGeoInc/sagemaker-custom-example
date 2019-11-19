FROM tensorflow/tensorflow:latest-py3
COPY linear.py .
ENTRYPOINT ["python", "linear.py"]
