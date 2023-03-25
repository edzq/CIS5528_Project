# Named Entity Extraction for BioMedical text

This is the project repository of CIS5528 Project in 2023 Spring, Temple University.

Team Members: Qi Zhang, Yihan Zhang, Xinwen Zhang

## Task Description

## Dataset

We are going to use three kinds of datasets: BC2GM, BC5CDR and NCBI. Recommeding the HuggingFace Dataset Library to load these copora (Click the following dataset name can redict to the HuggingFace dataset page).

1. [BC2GM](https://huggingface.co/datasets/bc2gm_corpus)

2. [BC5CDR](https://huggingface.co/datasets/tner/bc5cdr)

3. [NCBI](https://huggingface.co/datasets/ncbi_disease)

To load these dataset you can try:


## Baselines

1. CRF

论文和一些其他资料:

> - McCallum, Andrew, and Wei Li. "Early results for named entity recognition with conditional random fields, feature induction and web-enhanced lexicons." (2003).

> - [CRF for NER](https://www.dominodatalab.com/blog/named-entity-recognition-ner-challenges-and-model)
> - [理解条件随机场](https://www.zhihu.com/question/35866596/answer/236886066)
> - [条件随机场NER](https://zhuanlan.zhihu.com/p/119254570)


2. BiLSTM

论文和一些其他资料:

> - Marginal Likelihood Training of BiLSTM-CRF for Biomedical Named Entity Recognition from Disjoint Label Sets
> - [NER using BiLSTM](https://towardsdatascience.com/named-entity-recognition-ner-using-keras-bidirectional-lstm-28cd3f301f54)
> - 

3. BERTs

4. GPTs



## Reference:

1. Tong, Yiqi, Yidong Chen, and Xiaodong Shi. "A multi-task approach for improving biomedical named entity recognition by incorporating multi-granularity information." Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021. 2021.

This is a multi-task approach for biomecial NER models. The main idea of this paper is leveraging multi-level feature of Sequence from both token-level and sequence-level.




