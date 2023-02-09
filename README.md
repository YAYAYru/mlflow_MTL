# mlflow_MTL
Find the article on how to use Multitask learning with independent models using MLflow [Here](https://habr.com/ru/post/712904/) in Russian
## Quick start on the following sequential items.
- Tested in Ubuntu 20.08
- Init
```bash
git clone https://github.com/YAYAYru/paper_mlflow.git
cd paper_mlflow
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip install -r requirements.txt
```
- Run `dvc` pipeline. If you need to train then enable the GPU `export CUDA_VISIBLE_DEVICES='0'` else will slow train.
```bash
cd src
dvc repro
```
- First disable the GPU `export CUDA_VISIBLE_DEVICES=''` if there is one.  Then run two models with different ports `mlflow serve --no-conda -m <artifact_location>/<uuid>/artifacts/model -h 0.0.0.0 -p 8001`, example:
```bash
# for model1.py
mlflow models serve --no-conda -m file:///home/yayay/yayay/git/github/paper_mlflow/src/mlruns/902157297686484746/dcfc070aae044571af6577fa8f2f88b2/artifacts/model -h 0.0.0.0 -p 8001
# for model2.py
mlflow models serve --no-conda -m file:///home/yayay/yayay/git/github/paper_mlflow/src/mlruns/137049049665508372/b75e8ca4891d41e486e041fc996829e9/artifacts/model -h 0.0.0.0 -p 8002
```
You can also take `<artifact_location>/<uuid>/artifacts/model` from `mlflow ui`, where in the selected experiment ID you can find the inscription `Full Path:`. 
- See request and response result `src/request.py` or `notebooks/request.ipynb` if installed jupyter
```bash
python3 request.py
```
or
```bash
jupyter notebook
# choose notebooks/request.ipynb and run all
```

## Others
- Kill `mlflow`
```bash
ps -fA | grep mlflow
kill -9 last 4023 #(id process example)
```
