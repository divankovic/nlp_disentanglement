from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import Dataset



# TODO - update this to load BeerReviews and 20newsgroups, already preprocessed

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
        elif dataset == 'reuters':
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
    def load_20newsgroups(self, data_dir='resources/datasets/20_newsgroups_old'):
        splits = ['train', 'test']
        data = {}
        for split in splits:
            dataset = fetch_20newsgroups(data_home=data_dir, subset=split, remove=('headers', 'footers', 'quotes'))
            examples = []
            for (text, target) in zip(dataset.data, dataset.target):
                examples.append({'text': text, 'label': dataset.target_names[target]})
            data[split] = examples

        return data['train'], data['test']