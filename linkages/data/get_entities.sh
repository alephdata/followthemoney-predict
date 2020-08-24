#!/bin/bash

function paginate() {
    cur_url=$1
    while [ "$cur_url" != "null" ]; do
        data=$( curl -s "$cur_url" )
        echo $data
        cur_url=$( echo $data | jq -r '.next' )
    done;
}

now=$( date +%s )
paginate "${ALEPHCLIENT_HOST}/api/2/collections?api_key=${ALEPHCLIENT_API_KEY}" | jq -r '.results[].foreign_id' | (while read foreign_id; do
    echo "Getting foreign_id ${foreign_id}"
    foreign_id_fname=$( echo ${foreign_id} | tr ' ' '_' )
    alephclient stream-entities -s LegalEntity -f "${foreign_id}" | pv -l > raw/entities/legal_entity-${foreign_id_fname}-${now}.json
done; )

find entities/raw/ -size  0 -print -delete
