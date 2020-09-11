

data/xref.aleph.all.parquet:
    python -m followthemoney_predict.cli data --data-source aleph --output-file data/pairs.aleph.all.parquet --cache-dir ./cache/ xref

models/model.linear.pkl: data/xref.aleph.all.parquet
    python -m followthemoney_predict.cli model --data-file data/xref.aleph.all.parquet --output-file models/model.linear.pkl xref linear

models/model.xgboost.pkl: data/xref.aleph.all.parquet
    python -m followthemoney_predict.cli model --data-file data/xref.aleph.all.parquet --output-file models/model.xgboost.pkl xref xgboost
