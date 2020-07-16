import os
import shutil

import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from ststransformers.util.evaluation import pearson_corr, spearman_corr

from ststransformers.algo.transformer_model import STSTransformerModel

from examples.arabic_sts.config.transformer_config import TEMP_DIRECTORY, transformer_config, MODEL_TYPE, MODEL_NAME, \
    SEED
from examples.arabic_sts.normalizer import fit, un_fit
from examples.arabic_sts.reader import concatenate
import numpy as np

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

FILE_DIRECTORY = "data"
data = concatenate(FILE_DIRECTORY)
data = data[['text_a', 'text_b', 'labels']].dropna()

train, test = train_test_split(data, test_size=0.2)

test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

train = fit(train, 'labels')
test = fit(test, 'labels')

if transformer_config["evaluate_during_training"]:
    if transformer_config["n_fold"] > 1:
        test_preds = np.zeros((len(test), transformer_config["n_fold"]))
        for i in range(transformer_config["n_fold"]):

            if os.path.exists(transformer_config['output_dir']) and os.path.isdir(transformer_config['output_dir']):
                shutil.rmtree(transformer_config['output_dir'])

            model = STSTransformerModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                               args=transformer_config)
            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
            model.train_model(train_df, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                              mae=mean_absolute_error)
            model = STSTransformerModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=1,
                               use_cuda=torch.cuda.is_available(), args=transformer_config)

            predictions, raw_outputs = model.predict(test_sentence_pairs)
            test_preds[:, i] = predictions

        test['predictions'] = test_preds.mean(axis=1)

    else:
        model = STSTransformerModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                           args=transformer_config)
        train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
        model.train_model(train_df, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                          mae=mean_absolute_error)
        model = STSTransformerModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=1,
                           use_cuda=torch.cuda.is_available(), args=transformer_config)

        predictions, raw_outputs = model.predict(test_sentence_pairs)
        test['predictions'] = predictions

else:
    model = STSTransformerModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                       args=transformer_config)
    model.train_model(train, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)
    predictions, raw_outputs = model.predict(test_sentence_pairs)
    test['predictions'] = predictions


test = un_fit(test, 'predictions')

draw_scatterplot(dev, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), "English-German")
print_stat(test, 'labels', 'predictions')
