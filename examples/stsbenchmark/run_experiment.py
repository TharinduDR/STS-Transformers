import torch
from sklearn.metrics import mean_absolute_error
import pandas as pd

from examples.stsbenchmark.config.transformer_config import MODEL_TYPE, MODEL_NAME, transformer_config
from ststransformers.util.evaluation import pearson_corr, spearman_corr
from ststransformers.util.normalizer import fit
from ststransformers.algo.transformer_model import STSTransformerModel

train = pd.read_csv("examples/stsbenchmark/data/stsbenchmark/sts-train-dev.csv", usecols=[4, 5, 6], names=['labels', 'text_a', 'text_b'], sep='\t', engine="python", quotechar='"', error_bad_lines=False)
eval_df = pd.read_csv("examples/stsbenchmark/data/stsbenchmark/sts-test.csv", usecols=[4, 5, 6], names=['labels', 'text_a', 'text_b'], sep='\t', engine="python", quotechar='"', error_bad_lines=False)

train = fit(train, "labels")
eval_df = fit(eval_df, "labels")


model = STSTransformerModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                            args=transformer_config)


model.train_model(train, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                          mae=mean_absolute_error)
model = STSTransformerModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=1,
                            use_cuda=torch.cuda.is_available(), args=transformer_config)
result, model_outputs, wrong_predictions = model.eval_model(eval_df, pearson_corr=pearson_corr,
                                                                    spearman_corr=spearman_corr,
                                                                    mae=mean_absolute_error)
