import os
import pandas as pd

keywords = ['violencia', 'violencia de genero']

# TO COMPLETE
login_info = {"mc_session": "",
              "mc_remember_token": ""}

# Only latam
df_tags_country_latam = pd.read_csv("collections.csv")
df_tags_country_latam.index = df_tags_country_latam["Municipality code"]
collections_dic = df_tags_country_latam.to_dict()["tags_id"]
collections = list(collections_dic.keys())

start_date = '2017-12-31'
end_date = '2023-01-01'



import mediacloud.api, json, datetime

mc = mediacloud.api.MediaCloud(login_info['mc_remember_token'])


stories = []
last_processed_stories_id = 0
while len(stories) < 5000:
    fetched_stories = mc.storyList(f'{keywords[0]} AND media_id:{collections_dic[collections[0]]}', 
                                   solr_filter=mc.dates_as_query_clause(datetime.date(2018,1,1), datetime.date(2019,1,1)),
                                   last_processed_stories_id=last_processed_stories_id)
    stories.extend(fetched_stories)
    if len( fetched_stories) < fetch_size:
        break
    last_processed_stories_id = stories[-1]['processed_stories_id']
print(json.dumps(stories))