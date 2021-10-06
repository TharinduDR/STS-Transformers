from transformers import XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST, XLMRobertaConfig

from ststransformers.models.roberta_model import RobertaForSequenceClassification


class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
