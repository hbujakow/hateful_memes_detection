### In every script set variable `MLFLOW_TRACKING_URI`:

```python
MLFLOW_TRACKING_URI = "/home2/faculty/mgalkowski/memes_analysis/mlflow_data"
```

### then set tracking uri and  experiment (in the script) to:

```python
import mlflow

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("hateful_memes")
```

### then you can use mlflow in normal way e.g.

```python
with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
```

### it will log all the data to mlflow server if you enable autologging e.g. for sklearn:

```python
mlflow.sklearn.autolog()
```

### you can log manually differet metrics, parameters, artifacts (e.g. [here](https://mlflow.org/docs/latest/tracking.html#logging-functions))

------------------------------------------

## Running server and UI (to check experiments / models / runs):

```bash
mlflow ui --backend-store-uri /home2/faculty/mgalkowski/memes_analysis/mlflow_data --host 0.0.0.0 --port 5000
```

then create tunnel

```bash
ssh -NL 8888:dgx-4:5000 eden.mini.pw.edu.pl
```

and open in browser:

**http://localhost:5000**