#!/bin/bash
export API_TOKEN=6065972502:AAEidJXRtnouJT27vL0npxM4Mr34PkqAGI0
export CHAT_ID=@rockysuck_stock_channel

cd /Users/lmshek/Documents/GitHubExternal/stock ; /usr/bin/env /Users/lmshek/opt/anaconda3/envs/testing/bin/python backtester.py $(date -v -2d '+%Y-%m-%d') $(date -v -1d '+%Y-%m-%d')