#!/bin/sh
if [ -z "$1" ]; then echo "Specify interval in ms"; else fping -l -p $1 10.10.0.5; fi

