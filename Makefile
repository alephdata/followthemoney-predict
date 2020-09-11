
VOLUMES=-v ${PWD}/data:/data -v ${PWD}/cache:/cache -v ${PWD}/models:/models
FOllOWTHEMONEY_PREDICT=docker run ${VOLUMES} followthemoney-predict:latest

shell:
	docker run -it ${VOLUMES} --entrypoint /bin/bash followthemoney-predict:latest

data/xref.aleph.all.parquet:
	${FOllOWTHEMONEY_PREDICT} data --data-source aleph --output-file /data/xref.aleph.all.parquet --cache-dir /cache/ xref

models/model.linear.pkl: data/xref.aleph.all.parquet
	${FOllOWTHEMONEY_PREDICT} model --data-file /data/xref.aleph.all.parquet --output-file /models/model.linear.pkl xref linear

models/model.xgboost.pkl: data/xref.aleph.all.parquet
	${FOllOWTHEMONEY_PREDICT} model --data-file /data/xref.aleph.all.parquet --output-file /models/model.xgboost.pkl xref xgboost
