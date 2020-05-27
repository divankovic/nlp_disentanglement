from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import Dataset
from preprocess.text_preprocessing import clean_text, tokenize


# contains loader for different datasets used in experiments
# setup 20NewsGroups, IMDB first

class SimpleTextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DatasetLoader:

    def load_dateset(self, dataset):
        """
        Parameters
        ----------
        dataset - dataset name

        Returns
        -------
        dataset - dict{train:Dataset, test:Dataset}
        """
        if dataset == '20newsgroups':
            return self.load_20newsgroups()
        elif dataset == 'imdb':
            pass
        elif dataset == 'beerreviews':
            pass
        elif dataset == 'mnist':
            # just for testing purposes
            pass

    # 20NewsGroups
    # The dataset contains 20000 messages collected from 20 different
    # Usenet newsgroups (1000 messages from each group):
    #
    # | alt.atheism           | soc.religion.christian   | comp.windows.x     | sci.crypt
    # | talk.politics.guns    | comp.sys.ibm.pc.hardware | rec.autos          | sci.electronics
    # | talk.politics.mideast | comp.graphics            | rec.motorcycles    | sci.space
    # | talk.politics.misc    | comp.os.ms-windows.misc  | rec.sport.baseball | sci.med
    # | talk.religion.misc    | comp.sys.mac.hardware    | rec.sport.hockey   | misc.forsale
    def load_20newsgroups(self, data_dir='resources/datasets/20_newsgroups', preprocess=False):
        splits = ['train', 'test']
        data = {}
        for split in splits:
            dataset = fetch_20newsgroups(data_home=data_dir, subset=split, remove=('headers', 'footers', 'quotes'))
            examples = []
            for (text, target) in zip(dataset.data, dataset.target):
                examples.append({'text': text, 'label': dataset.target_names[target]})
            data[split] = examples

        if preprocess:
            self.custom_20newsgroups_preprocess(data)

        return data['train'], data['test']

    def custom_20newsgroups_preprocess(self, data):
        """
            Custom preprocessing method based on the implementation from 'Structured disentangled representations' of
            HFVAE on 20Newsgroups.
        """
