#!/bin/bash
cd /Users/lmshek/Documents/GitHubExternal/stock
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

/usr/bin/env /Users/lmshek/opt/anaconda3/envs/testing/bin/python backtester.py $(date -v -2d '+%Y-%m-%d') $(date -v -1d '+%Y-%m-%d')