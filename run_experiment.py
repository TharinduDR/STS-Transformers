import torch
from sklearn.metrics import mean_absolute_error
import pandas as pd
from config.transformer_config import transformer_config, MODEL_TYPE, MODEL_NAME
from evaluation import pearson_corr, spearman_corr
from normalizer import fit
from transformer_model import TrnsformerModel


train = pd.read_csv("data/stsbenchmark/sts-train-dev.csv", usecols=[4, 5, 6], names=['labels', 'text_a', 'text_b'])
eval_df = pd.read_csv("data/stsbenchmark/sts-test.csv", usecols=[4, 5, 6], names=['labels', 'text_a', 'text_b'])

train = fit(train, "labels")
eval_df = fit(eval_df, "labels")

model = TrnsformerModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                           args=transformer_config)


model.train_model(train, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                          mae=mean_absolute_error)
model = TrnsformerModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=1,
                           use_cuda=torch.cuda.is_available(), args=transformer_config)
result, model_outputs, wrong_predictions = model.eval_model(eval_df, pearson_corr=pearson_corr,
                                                                    spearman_corr=spearman_corr,
                                                                    mae=mean_absolute_error)
