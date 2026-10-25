# Python and package versions match requirements.txt, the environment that
# serving/train_model.py pickles model.pkl with. A pickled scikit-learn model
# is not guaranteed to load in a different scikit-learn/numpy version.
FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.139.0 \
    uvicorn[standard]==0.51.0 \
    pydantic==2.12.5 \
    scikit-learn==1.8.0 \
    numpy==2.4.2 \
    pandas==2.3.3

COPY app.py model.pkl feature_cols.json ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
