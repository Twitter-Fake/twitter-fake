


"""
    You should create a file named "tweet_conf.json" in the same directory to this code to work. 
    Template is as shown in below.

    {
    "consumer_key": "*********************************",
    "consumer_secret": "*************************************", 
    "access_key": "*******************************************",
    "access_secret": "***************************************"
    }

    These information can be generated from the twitter developer page.



"""


import tweepy
import json
"""[summary]

Returns:
    user token
"""
import os
def get_conf():
    folder = os.path.dirname(__file__)
    with open( os.path.join(folder,'tweet_conf.json')) as f:
        return json.load(f)

def get_dict(user):
    dict_mapping = {}
    for key in user.__dict__:
        dict_mapping[key] = getattr(user, key)
    return dict_mapping


cnt = -1
auth_obj = None
def autheniticate():
    global cnt
    global auth_obj 
    cnt += 1
    
    # if(cnt % 10 != 0):
    #     return auth_obj

    user_conf = get_conf()
    auth_obj = tweepy.OAuthHandler(user_conf['consumer_key'], user_conf['consumer_secret'])
    auth_obj.set_access_token(user_conf['access_key'], user_conf['access_secret'])
    return auth_obj
"""

'id', 'name', 'screen_name', 'statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'url', 'lang', 'time_zone', 'location', 'default_profile', 'default_profile_image', 'geo_enabled', 'profile_image_url', 'profile_banner_url', 'profile_use_background_image', 'profile_background_image_url_https', 'profile_text_color', 'profile_image_url_https', 'profile_sidebar_border_color', 'profile_background_tile', 'profile_sidebar_fill_color', 'profile_background_image_url', 'profile_background_color', 'profile_link_color', 'utc_offset', 'protected', 'verified', 'description', 'created_at', 'updated', 'file', 'label', 'tweet']
"""

headers = ['id', 'name', 'screen_name', 'statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'url', 'lang', 'time_zone', 'location', 'default_profile', 'default_profile_image', 'geo_enabled', 'profile_image_url', 'profile_banner_url', 'profile_use_background_image', 'profile_background_image_url_https', 'profile_text_color', 'profile_image_url_https', 'profile_sidebar_border_color', 'profile_background_tile', 'profile_sidebar_fill_color', 'profile_background_image_url', 'profile_background_color', 'profile_link_color', 'utc_offset', 'protected', 'verified', 'description', 'created_at', 'updated', 'file', 'label', 'tweet']


header_set = set(headers)

def filter_props(user_dict):
    global header_set
    new_dict = {}

    for key, value in user_dict.items():
        if key in header_set:
            new_dict[key] = value
    
    return new_dict


def get_user_info(id):
    auth = autheniticate()
    api = tweepy.API(auth, wait_on_rate_limit = True )

    number_of_tweets=200
    tweets = api.user_timeline(id = id)
 
    user_obj = api.get_user(id=id)
    user_dict = get_dict(user_obj)
    user_dict['tweet'] = ""

    for tweet in tweets:
        user_dict['tweet'] += tweet.text +" "
     
    filter_user_dict = filter_props(user_dict)
    return filter_user_dict


def write_out(df_cur, working_list, out_put_file):
    
    if df_cur.shape[0] == 0:
        df_cur = pd.DataFrame(working_list, columns = headers )
    elif len(working_list) != 0:
        df_new = pd.DataFrame(working_list, columns = headers )
        df_cur = pd.concat([df_cur, df_new] )
    
    df_cur.to_csv(out_put_file, index= False)

    return df_cur

"""
    download all
"""
import pandas as pd
import csv 
def download_all(input_file, out_put_file):
    global headers
    df_cur = pd.DataFrame(columns=headers)
    completed = set()
    if os.path.exists(out_put_file):
        df_cur = pd.read_csv(out_put_file)
        completed = set(df_cur['id'].values.tolist())
    

    working_list = []

    with open(input_file) as f:
        csv_reader = csv.reader(f,  delimiter='\t')
        #headers = next(csv_reader)
        counter = 0
        err = 0
        for row in csv_reader:
            if int(row[0]) not in completed:
                """
                    catch user not exist
                """
                completed.add(int(row[0]))
                try:
                    temp_dict = get_user_info(row[0])
                    temp_dict['label'] = 0 if row[1]=='human' else 1
                    working_list.append(temp_dict)
                    #break
                    counter += 1
                    if(counter % 50 == 0):
                        print('Download as of now {}'.format(counter))
                        ## write
                        df_cur = write_out(df_cur, working_list, out_put_file)
                        working_list = []
                except Exception as e:
                    err += 1
                    if(err % 10 == 0):
                        print('Error count= {}, reason = {}, id = {}'.format(err, e, row[0]))
                    pass


        write_out(df_cur, working_list, out_put_file)

    
def download_all_folder(in_folder, out_file):

    for file in os.listdir(in_folder):
        file = os.path.join(in_folder, file)
        download_all(file, out_file)


download_all_folder('../data/new_temp', '../data/new_data.csv')

