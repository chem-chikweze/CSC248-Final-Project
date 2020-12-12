import sys
from pathlib import Path

from torch.nn import CrossEntropyLoss

from finbert.finBERT.finbert import finbert
from finbert.finBERT.finbert.finbert import Config, FinBert

sys.path.append('..')
import shutil
import pandas as pd

from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import *

from sklearn.metrics import classification_report


def report(df, cols=['label', 'prediction', 'logits']):
    # print('Validation loss:{0:.2f}'.format(metrics['best_validation_loss']))
    cs = CrossEntropyLoss(weight=finbert.class_weights)
    loss = cs(torch.tensor(list(df[cols[2]])), torch.tensor(list(df[cols[0]])))
    print("Loss:{0:.2f}".format(loss))
    print("Accuracy:{0:.2f}".format((df[cols[0]] == df[cols[1]]).sum() / df.shape[0]))
    print("\nClassification Report:")
    print(classification_report(df[cols[0]], df[cols[1]]))


def train_bert():
    project_dir = str(Path.cwd()) + "/finbert/finBERT"
    project_dir = str(Path.cwd()) + "/finbert/finBERT"
    pd.set_option('max_colwidth', -1)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR)

    lm_path = project_dir / 'models' / 'language_model' / 'finbertTRC2'
    cl_path = project_dir / 'models' / 'classifier_model' / 'finbert-sentiment'
    cl_data_path = project_dir / 'data' / 'sentiment_data'

    # Clean the cl_path
    try:
        shutil.rmtree(cl_path)
    except:
        pass

    bertmodel = BertForSequenceClassification.from_pretrained(lm_path, cache_dir=None, num_labels=3)

    config = Config(data_dir=cl_data_path,
                    bert_model=bertmodel,
                    num_train_epochs=4,
                    model_dir=cl_path,
                    max_seq_length=48,
                    train_batch_size=32,
                    learning_rate=2e-5,
                    output_mode='classification',
                    warm_up_proportion=0.2,
                    local_rank=-1,
                    discriminate=True,
                    gradual_unfreeze=True)

    print("Fine tuning BERT model to the financial domain!\n")
    finbert = FinBert(config)
    finbert.prepare_model(label_list=['positive', 'negative', 'neutral'])

    # Get the training examples
    train_data = finbert.get_data('train')
    model = finbert.create_the_model()

    trained_model = finbert.train(train_examples=train_data, model=model)

    test_data = finbert.get_data('test')

    results = finbert.evaluate(examples=test_data, model=trained_model)

    results['prediction'] = results.predictions.apply(lambda x: pd.np.argmax(x, axis=0))

    report(results, cols=['labels', 'prediction', 'predictions'])
