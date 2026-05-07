FROM python:3.10-slim

WORKDIR /app

# Installer Java + dépendances système (requis par PySpark)
RUN apt-get update && apt-get install -y \
    curl \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Définir JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
