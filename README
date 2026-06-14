# DermAI — Clasificador Dermatológico (TFG)

Sistema de clasificación automática de enfermedades dermatológicas mediante Deep Learning, desarrollado como Trabajo de Fin de Grado en Ingeniería Informática (ICAI / Universidad de Málaga).

> ⚠️ **Aviso:** Esta aplicación tiene fines exclusivamente educativos e informativos. Los resultados ofrecidos son orientativos y no constituyen diagnóstico médico. Ante cualquier afección cutánea, consulte siempre con un dermatólogo o profesional sanitario.

## Descripción

La aplicación permite cargar una imagen de una lesión cutánea y obtener una predicción sobre la patología más probable, utilizando modelos EfficientNet-B0 entrenados sobre el dataset [DermNet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet).

Se ofrecen dos configuraciones de modelo, entre las que el usuario puede alternar:

- **Modelo de 10 clases**: entrenado con las 10 clases mayoritarias del dataset.
- **Modelo de 23 clases**: entrenado con las 23 clases originales del dataset.

## Arquitectura

- **Backend**: FastAPI + PyTorch (`main.py`)
- **Frontend**: HTML, CSS y JavaScript estáticos (`static/`)

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/jxzhuu/tfg-clasificacion-dermatologica.git
   cd tfg-clasificacion-dermatologica
   ```

2. Crea un entorno virtual e instala las dependencias:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

3. Descarga los pesos de los modelos (`efficientnetb0_10clases.pth` y `efficientnetb0_23clases.pth`) y colócalos en la ruta indicada en `main.py`.

## Ejecución

```bash
uvicorn main:app --reload
```

La aplicación estará disponible en [http://localhost:8000](http://localhost:8000).

## Estructura del proyecto

```
app-web-modelo/
├── main.py              # Backend FastAPI
├── static/
│   └── index.html       # Frontend
├── requirements.txt
└── README.md
```

## Autor

Jiaxing Zhu — Grado en Ingeniería Informática, Universidad de Málaga
Tutor: Ezequiel López Rubio
