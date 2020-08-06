#!/bin/bash

DATE=$( date +%Y%m%d%H%M%S )
DEFAULT_FILENAME="linkages/linkages-$DATE.json"
if [ -z "$1" ]; then
    FILENAME=$DEFAULT_FILENAME;
else
    FILENAME=$1
fi

if [ -z "$ALEPHCLIENT_HOST" ]; then
    export ALEPHCLIENT_HOST=https://aleph.occrp.org/
fi

mkdir ./linkages/ 2> /dev/null
alephclient linkages | pv -l > $FILENAME
