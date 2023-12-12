# ProxyRCA

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/proxy-based-item-representation-for-attribute/recommendation-systems-on-amazon-fashion)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-fashion?p=proxy-based-item-representation-for-attribute)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/proxy-based-item-representation-for-attribute/recommendation-systems-on-amazon-men)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-men?p=proxy-based-item-representation-for-attribute)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/proxy-based-item-representation-for-attribute/recommendation-systems-on-amazon-beauty)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-beauty?p=proxy-based-item-representation-for-attribute)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/proxy-based-item-representation-for-attribute/recommendation-systems-on-amazon-games)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-games?p=proxy-based-item-representation-for-attribute)

Implementation of the paper **"Proxy-based Item Representation for Attribute and Context-aware Recommendation"**, accepted in *The 17th ACM International Conference on Web Search and Data Mining* (WSDM 2024) \[[arXiv link](https://arxiv.org/abs/2312.06145)\].

---

* raw dataset directory: `./raw/`
    * put [CARCA/Data](https://github.com/ahmedrashed-ml/CARCA) as `./raw/CARCA/`
    * `./raw/ml1m`
    * `./raw/ml20m`
* data directory: `./data/`

For preprocessing, anaconda environment with `requirements.txt` installed is recommended.

```bash
python preprocess.py prepare --dname ml1m
python preprocess.py prepare --dname ml20m
python preprocess.py prepare --dname fashion
python preprocess.py prepare --dname beauty
python preprocess.py prepare --dname men
python preprocess.py prepare --dname game

python preprocess.py split_quarters --dname fashion

python preprocess.py count_stats
```

For GPU runs, first build docker:

```bash
./scripts/build.sh
```

Then, use the following (dockerized python run):

```bash
runpy () {
    docker run \
        -it \
        --rm \
        --init \
        --gpus '"device=0"' \
        --shm-size 32G \
        --volume="$HOME/.cache/torch:/root/.cache/torch" \
        --volume="$PWD:/workspace" \
        proxyrca \
        python "$@"
}

runpy entry.py fashion/proxyrca
```

Easy tensorboard:

```bash
./scripts/tboard.sh
```
