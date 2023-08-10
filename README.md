# ProxyRCA

*anonymized implementation for WSDM 2024 submission*

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
