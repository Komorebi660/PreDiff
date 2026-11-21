# Process Movielens, Beauty, Toys data
import pickle
import numpy as np
import random
import json
from collections import defaultdict
import argparse


def load_interaction(data_dir):
    all_data = []
    f = open(data_dir, 'r')
    if 'ml-25m' in data_dir:
        f.readline()    # ignore header
    user2id, item2id = {}, {}
    for eachline in f:
        if 'ml-1m' in data_dir:
            eachline = eachline.split('::')
            u, i, r, t = eachline[0], eachline[1], eachline[2], eachline[3]
        elif 'ml-25m' in data_dir:
            eachline = eachline.split(',')
            u, i, r, t = eachline[0], eachline[1], eachline[2], eachline[3]
        else:
            eachline = json.loads(eachline)
            u, i, t = eachline['reviewerID'], eachline['asin'], eachline['unixReviewTime']
        
        if u not in user2id:
            user2id[u] = len(user2id) + 1  # encode from 1
        user = user2id[u]
        
        if i not in item2id:
            item2id[i] = len(item2id) + 1  # encode from 1
        item = item2id[i]
        
        all_data.append((user, item, t))
    f.close()
    
    print('user num:', len(user2id))
    print('item num:', len(item2id))
    print('interaction num:', len(all_data))
    
    user_seq = defaultdict(list)
    for user, item, time in all_data:
        user_seq[user].append((item, time))         # {user_id: [(item_id, time), ...], ...}
    
    for user, seq in user_seq.items():
        seq = sorted(seq, key=lambda x: x[1])       # sorted by time
        user_seq[user] = [item for item, _ in seq]  # {user_id: [item_id, ...], ...}
    
    print("avg. interaction per user:", np.mean([len(x) for x in user_seq.values()]))
    
    return user_seq, item2id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training args.')
    parser.add_argument('--dataset', type=str, default='beauty', 
                        choices=['beauty', 'toys', 'movielens', 'movielens-25m'], help='name of dataset')
    parser.add_argument('--seq_len', type=int, default=50, help='max length of sequence')
    args = parser.parse_args()

    random.seed(2024)
    np.random.seed(2024)

    if args.dataset == 'beauty':
        f_interation = './reviews_Beauty_5.json'
    elif args.dataset == 'toys':
        f_interation = './reviews_Toys_and_Games_5.json'
    elif args.dataset == 'movielens':
        f_interation = './ml-1m/ratings.dat'
    elif args.dataset == 'movielens-25m':
        f_interation = './ml-25m/ratings.csv'
    else:
        raise ValueError('Invalid dataset name')

    user_seq, item2id = load_interaction(f_interation)

    with open(f"./{args.dataset}.pickle", 'wb') as f:
        pickle.dump((len(item2id), args.seq_len, user_seq), f)
