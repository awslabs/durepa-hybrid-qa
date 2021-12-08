# DuRePa: Dual Reader-Parser on Hybrid Textual and Tabular Evidence for Open Domain Question Answering

Code and model from our [ACL 2021 paper](https://arxiv.org/abs/2108.02866).

## Abstract
The current state-of-the-art generative models for open-domain question answering (ODQA) have focused on generating direct answers from unstructured textual information. However, a large amount of world's knowledge is stored in structured databases, and need to be accessed using query languages such as SQL. Furthermore, query languages can answer questions that require complex reasoning, as well as offering full explainability. In this paper, we propose a hybrid framework that takes both textual and tabular evidence as input and generates either direct answers or SQL queries depending on which form could better answer the question. The generated SQL queries can then be executed on the associated databases to obtain the final answers. To the best of our knowledge, this is the first paper that applies Text2SQL to ODQA tasks. Empirically, we demonstrate that on several ODQA datasets, the hybrid methods consistently outperforms the baseline models that only take homogeneous input by a large margin. Specifically we achieve state-of-the-art performance on OpenSQuAD dataset using a T5-base model. In a detailed analysis, we demonstrate that the being able to generate structural SQL queries can always bring gains, especially for those questions that requires complex reasoning. 

## Setup
```
conda create --name durepa python=3.7
source activate durepa
conda install pytorch=1.6 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

## Train model
```
python run_ranking.py
```

## Inference
```
python run_inference.py
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

