FROM python:3.11

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "app.py"]
