cd /Users/lmshek/Documents/GitHubExternal/stock
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

/usr/bin/env /Users/lmshek/opt/anaconda3/envs/testing/bin/python stockfinder.py