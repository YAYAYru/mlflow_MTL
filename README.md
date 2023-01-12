# paper_mlflow

- Init
```bash
pip install -r requirements.txt
python src/model1.py
python src/model2.py
mlflow server --backend-store-uri file:src/mlruns --no-serve-artifacts
```

- Убрать процесс `mlflow`
```bash
ps -fA | grep mlflow
kill -9 1005473
```

