# OneBlue — Adriatic Sea Current Prediction

## Prerequisites

- Python 3.13+
- [Copernicus Marine Service](https://marine.copernicus.eu) account (free)

## Installation

```bash
git clone <repo-url>
cd oneblue
pip install -e .
```

## Configuration

Create `secrets.yaml` in the project root:

```yaml
cmems:
  username: "your_username"
  password: "your_password"
```

Model and data settings are in `config.yaml`.

## Usage

```bash
# Verify credentials and test data download
python -m src.main copernicus test-download

# Download data
python -m src.main copernicus download-specific <dataset_id> <variable>

# Train models
python -m src.main convlstm train
python -m src.main lstm point-by-point

# Run baselines
python -m src.main baselines mean-nday
```

Pass `--help` to any command for options.
