# STSTransformers : Transformer based Semantic Textual Similrity. 

STSTransformers provides state-of-the-art models for Semantic Textual Similarity.

## Installation
you first need to install PyTorch.
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When PyTorch has been installed, you can install from source by cloning the repository and running:

```bash
git clone https://github.com/TharinduDR/STS-Transformers.git
cd TransQuest
pip install -r requirements.txt
``` 

## Pretrained Models

### Arabic
We have released four STSTransformer models Arabic with and without pre-segmentation based on multilingual bert and Arabert. The models were evaluated using [STS 2017 Arabic monolingual dataset](http://alt.qcri.org/semeval2017/task1/).

| Transformer Type                         | Preprocess Pipeline                | Pearson Correlation | Spearman Correlation | Link                                |
| ---------------------------------------- |-----------------------------------:| -------------------:| --------------------:| ------------------------------------|
| Arabert                                  | Farasa Segmentation                | 0.7643              | 0.7873               | [model.zip](https://bit.ly/2P5f6nM) |
| Arabert                                  | None                               | 0.7032              | 0.7230               | [model.zip](https://bit.ly/2P5f6nM) |
| bert-base-multilingual-cased             | Farasa Segmentation                | 0.6766              | 0.6970               | [model.zip](https://bit.ly/2P5f6nM) |
| bert-base-multilingual-cased             | None                               | 0.6529              | 0.6706               | [model.zip](https://bit.ly/2P5f6nM) |