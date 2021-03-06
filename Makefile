VOLUMES=-v ${PWD}/data:/data -v ${PWD}/cache:/cache -v ${PWD}/models:/models -v ${PWD}/followthemoney_predict:/usr/src/app/followthemoney_predict
FOllOWTHEMONEY_PREDICT=docker run ${VOLUMES} followthemoney-predict:latest

.PHONY: shell predict publish build

all: evaluate

publish: clean
	python setup.py sdist bdist_wheel
	python3 -m twine upload dist/*

clean:
	rm -rf followthemoney_predict.egg-info build dist

build:
	docker build -t followthemoney-predict:latest .

shell:
	docker run -it ${VOLUMES} --entrypoint /bin/bash followthemoney-predict:latest

data/xref.%.all.parquet:
	${FOllOWTHEMONEY_PREDICT} \
		data \
			--data-source "$*" \
			--workflow "dask" \
			--dask-nworkers 6 \
			--dask-threads-per-worker 8 \
			--aleph-host ${ALEPHCLIENT_HOST} \
			--aleph-api-key ${ALEPHCLIENT_API_KEY} \
			--output-file /data/xref.$*.all.parquet \
			--cache-dir /cache/ \
		xref

models/model.%.pkl: data/xref.aleph.all.parquet
	${FOllOWTHEMONEY_PREDICT} \
		model \
			--data-file /data/xref.aleph.all.parquet \
			--output-file /models/model.$*.pkl \
		xref \
			--best-of 10 \
		$*

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
