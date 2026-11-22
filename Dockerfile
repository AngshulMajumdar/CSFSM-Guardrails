FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip install --no-cache-dir '.[api]' && useradd --uid 1000 --create-home csfsm && mkdir /data && chown csfsm:csfsm /data
USER csfsm
ENV CSFSM_DATABASE=/data/csfsm.sqlite3
EXPOSE 8000
CMD ["csfsm", "serve", "--host", "0.0.0.0", "--port", "8000"]
