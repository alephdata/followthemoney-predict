#!/bin/bash

collection_foreign_ids=$( cut -d# -f1 ./collection_foreign_ids.txt | tr -d "[:blank:]" )
for foreign_id in $collection_foreign_ids; do
    echo "Streaming entities from ${foreign_id}"
    alephclient stream-entities --publisher --foreign-id ${foreign_id} | pv -l > ./collections/${foreign_id}.json
    echo $collection
done
