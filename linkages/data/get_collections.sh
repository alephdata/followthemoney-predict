#!/bin/bash

find collections/ -size 0 -delete

collection_foreign_ids=$( cut -d# -f1 ./collection_foreign_ids.txt | tr -d "[:blank:]" )
for foreign_id in $collection_foreign_ids; do
    outname="./collections/${foreign_id}.json"
    if [ ! -e "$outname" ]; then
        echo "Streaming entities from ${foreign_id}"
        alephclient stream-entities --publisher --foreign-id ${foreign_id} | pv -l > $outname
    fi
done
