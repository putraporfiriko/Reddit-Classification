import praw
import json
import os

localarray = []
blankdict = {}

subredditname = str(input("Enter subreddit name: "))
timeframe = str(input("Enter timeframe (hour, day, week, month, year, all): "))
postlimit = int(input("Enter post limit: "))
filename = f"{subredditname}_{timeframe}_{postlimit}"
reddit = praw.Reddit(client_id='_f9-t1ynLC5hm5AgJnDWSQ',
                     client_secret='qV_00B7Gp4e5fLZSz4tb5zd4Nt_iHg',
                     user_agent='basdat')

hotpost = reddit.subreddit(subredditname).top(time_filter = timeframe, limit = postlimit)


for post in hotpost:
    blankdict["posttitle"] = post.title
    blankdict['postflairs'] = post.link_flair_text
    localarray.append(blankdict.copy())


# Make sure the directories exist
os.makedirs('Successful JSON exports', exist_ok=True)
os.makedirs('Successful JSON exports/newline-delimited', exist_ok=True)

# export to .json file (pretty-printed format)
with open(f'Successful JSON exports/{filename}.json', 'w') as outfile:
    json.dump(localarray, outfile, indent=6)

# export to newline-delimited JSON file (more efficient for large datasets)
with open(f'Successful JSON exports/newline-delimited/nd_{filename}.json', 'w') as obj:
    for record in localarray:
        obj.write(json.dumps(record) + '\n')
