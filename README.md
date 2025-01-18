# FairMT-Bench

This repository contains the code and released models for our paper FairMT-Bench: Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs(https://arxiv.org/pdf/2410.19317). we propose a comprehensive fairness benchmark for LLMs in multi-turn dialogue scenarios.



# Quick Start

Here we provide a quick start on how to evaluate your model on FairMT-1K.


## Install Requirements


First, create a Python virtual environment using e.g. Conda:
```shell
conda create -n handbook python=3.10 && conda activate handbook
```


You can then install the remaining package as follows:

```shell
pip install -r requirements.txt
```


## Generate Responses on FairMT-1K


* Here is the code template for generate response of your model on FairMT-1K:
```shell
python generate_answer.py \
--model <your_model> --prompt prompt_standard.txt \
--dataset <data_split> --save_path <your_save_path>
```
* For example
```shell
python generate_answer.py \
--model "google/gemma-1.1-7b-it" --prompt prompt_standard.txt \
--dataset coreference.json --save_path data/coreference
```

## Evaluation

Here is the code template for evaluate the generated response of your model on FairMT-1K with GPT-4:
```shell
export OPENAI_API_KEY=<your_api>
python evaluation_coreference.py \
--model "gpt4" --prompt prompt_standard.txt \
--dataset data/coreference/<generated_file>.json --save_path data/coreference/evaluation```
```

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@article{fan2024fairmt,
  title={FairMT-Bench: Benchmarking Fairness for Multi-turn Dialogue in Conversational LLMs},
  author={Fan, Zhiting and Chen, Ruizhe and Hu, Tianxiang and Liu, Zuozhu},
  journal={arXiv preprint arXiv:2410.19317},
  year={2024}
}
```
