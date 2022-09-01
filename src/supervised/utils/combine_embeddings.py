import numpy as np
import argparse

## Simple script to concatenate train/validation/test embeddings into one embeddings file. 
# Note that naming convention is not particularly flexible atm!.. 

def combine_embeddings():
    species = directory.split('/')[-1]

    test = np.load(f"{directory}/{species}_test_embeddings.npz")
    train = np.load(f"{directory}/{species}_train_embeddings.npz")
    val = np.load(f"{directory}/{species}_validation_embeddings.npz")
    con = dict()
    con['embeddings'] = np.concatenate((train['embeddings'], val['embeddings'], test['embeddings']), axis=0)
    con['labels'] = np.concatenate((train['labels'], val['labels'], test['labels']), axis=0)

    train_size = np.shape(train['embeddings'])[0]
    val_size = np.shape(val['embeddings'])[0]

    con['splits'] = [train_size, train_size + val_size]

    np.savez(f"{directory}/{species}_combined", embeddings=con['embeddings'], labels=con['labels'], splits=con['splits'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parameters for visualising the embeddings via TSNE"
    )
    parser.add_argument(
        "--embeddings_directory",
        type=str,
        required=True,
        help="Path to directory containing the train/val/test embeddings files to combine",
    )
    args = parser.parse_args()
    directory = args.embeddings_directory
    
    combine_embeddings()
    

