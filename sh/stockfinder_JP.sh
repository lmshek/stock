cd /Users/lmshek/Documents/GitHubExternal/stock
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

cd /Users/lmshek/Documents/GitHubExternal/stock & /usr/bin/env /Users/lmshek/opt/anaconda3/envs/testing/bin/python stockfinder_technical_breakout.py --market JP --stock_list nikkei_225