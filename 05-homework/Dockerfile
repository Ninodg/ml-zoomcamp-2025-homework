FROM agrigorev/zoomcamp-model:2025

# El directorio de trabajo es desde donde se ejecuta el contenedor al iniciarse
WORKDIR /code

# Copiamos todos los archivos del build context al directorio /app del contenedor
COPY ./requirements.txt /code/requirements.txt
COPY ./pipeline_v1.bin /code/pipeline_v1.bin

# Ejecutamos pip para instalar las dependencias en el contenedor
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

# Ejecuta nuestra aplicación cuando se inicia el contenedor
CMD ["fastapi", "run", "app/main.py", "--port", "80"]