VOLUMES=-v ${PWD}/data:/data -v ${PWD}/cache:/cache -v ${PWD}/models:/models
FOllOWTHEMONEY_PREDICT=docker run ${VOLUMES} followthemoney-predict:latest

.PHONY: shell predict

build:
	docker build -t followthemoney-predict:latest .

shell:
	docker run -it ${VOLUMES} --entrypoint /bin/bash followthemoney-predict:latest

data/xref.aleph.all.parquet:
	${FOllOWTHEMONEY_PREDICT} \
		data 
			--data-source aleph \
			--aleph-host ${ALEPHCLIENT_HOST} \
			--aleph-api-key ${ALEPHCLIENT_API_KEY} \
			--output-file /data/xref.aleph.all.parquet \
			--cache-dir /cache/ \
		xref

models/model.linear.pkl: data/xref.aleph.all.parquet
	${FOllOWTHEMONEY_PREDICT} \
		model \
			--data-file /data/xref.aleph.all.parquet \
			--output-file /models/model.linear.pkl \
		xref \
		linear

models/model.xgboost.pkl: data/xref.aleph.all.parquet
	${FOllOWTHEMONEY_PREDICT} \
		model \
			--data-file /data/xref.aleph.all.parquet \
			--output-file /models/model.xgboost.pkl \
		xref \
		xgboost

evaluate: models/model.xgboost.pkl models/model.linear.pkl
	${FOllOWTHEMONEY_PREDICT} \
		evaluate \
			--cache-dir /cache/ \
			--aleph-host ${ALEPHCLIENT_HOST} \
			--aleph-api-key ${ALEPHCLIENT_API_KEY} \
			--model-file /models/model.linear.pkl \
		xref \
			-c zz_laundromat_azerbaijani \
			-e c6dd1f196ff0fc331c74643c6b409b5d1fd7a81b.7270b91ce2c5eeddfc925dcab9354a5cddb80ee8 \
			--summary
	${FOllOWTHEMONEY_PREDICT} \
		evaluate \
			--cache-dir /cache/ \
			--aleph-host ${ALEPHCLIENT_HOST} \
			--aleph-api-key ${ALEPHCLIENT_API_KEY} \
			--model-file /models/model.xgboost.pkl \
		xref \
			-c zz_laundromat_azerbaijani \
			-e c6dd1f196ff0fc331c74643c6b409b5d1fd7a81b.7270b91ce2c5eeddfc925dcab9354a5cddb80ee8 \
			--summary
