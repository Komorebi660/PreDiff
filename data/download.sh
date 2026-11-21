#!/bin/bash
set -e

usage() {
    echo "Usage: $0 <dataset> [dataset2 ...]"
    echo "  Datasets: beauty, toys, ml1m, ml25m"
    echo "  Example: $0 beauty toys"
    exit 1
}

[ $# -eq 0 ] && usage

for ds in "$@"; do
    case "$ds" in
        beauty)
            echo "=== Downloading Beauty ==="
            wget -nc http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz || true
            gzip -df reviews_Beauty_5.json.gz 2>/dev/null || true
            echo "=== Processing Beauty ==="
            python process.py --dataset beauty --seq_len 50
            rm -f reviews_Beauty_5.json
            echo "=== Beauty done ==="
            ;;
        toys)
            echo "=== Downloading Toys ==="
            wget -nc http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz || true
            gzip -df reviews_Toys_and_Games_5.json.gz 2>/dev/null || true
            echo "=== Processing Toys ==="
            python process.py --dataset toys --seq_len 50
            rm -f reviews_Toys_and_Games_5.json
            echo "=== Toys done ==="
            ;;
        ml1m)
            echo "=== Downloading ML-1M ==="
            wget -nc http://files.grouplens.org/datasets/movielens/ml-1m.zip || true
            python -c "import zipfile; zipfile.ZipFile('ml-1m.zip').extract('ml-1m/ratings.dat')"
            echo "=== Processing ML-1M ==="
            python process.py --dataset movielens --seq_len 200
            mv movielens.pickle ml1m.pickle
            rm -rf ml-1m ml-1m.zip
            echo "=== ML-1M done ==="
            ;;
        ml25m)
            echo "=== Downloading ML-25M ==="
            wget -nc https://files.grouplens.org/datasets/movielens/ml-25m.zip || true
            python -c "import zipfile; zipfile.ZipFile('ml-25m.zip').extract('ml-25m/ratings.csv')"
            echo "=== Processing ML-25M ==="
            python process.py --dataset movielens-25m --seq_len 200
            mv movielens-25m.pickle ml25m.pickle
            rm -rf ml-25m ml-25m.zip
            echo "=== ML-25M done ==="
            ;;
        *)
            echo "Unknown dataset: $ds"
            usage
            ;;
    esac
done
echo "All done!"
