cd /Users/lmshek/Documents/GitHubExternal/stock
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

cd /Users/lmshek/Documents/GitHubExternal/stock & /usr/bin/env /Users/lmshek/opt/anaconda3/envs/testing/bin/python stockfinder_technical_breakout.py --market US --stock_list nasdaq_100 s_and_p