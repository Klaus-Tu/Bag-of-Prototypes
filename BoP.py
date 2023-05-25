import os
import numpy as np
import pickle as pkl
from sklearn.cluster import KMeans

class BoP:
    def __init__(self, ref_data, n_prototype, cache_path):
        """
        :param ref_data: the features of samples from reference dataset to construct codebook
        :param n_prototype: number of prototypes in the codebook
        :param cache_path: the path to save the BoP file
        """
        self.n_prototype = n_prototype
        self.cache_path = cache_path
        self.ref_data = ref_data
        self.prototypes = None
        self.ref_bop = None
        self.results_file = None
        self.file_name = f'{n_prototype}_prototypes'

        os.makedirs(cache_path, exist_ok=True)

        bop_file = os.path.join(cache_path, self.file_name + '.pkl')
        if not os.path.isfile(bop_file):
            bop_file = os.path.join(cache_path, self.file_name + '.pkl')
            self.construct_codebook(ref_data, bop_file)
        else:
            self.__read_from_bop_file(bop_file)

    def construct_codebook(self, samples, bop_file):

        self.codebook = KMeans(n_clusters=self.n_prototype, max_iter=100, random_state=2022).fit(samples)

        label_vals, label_counts = np.unique(self.codebook.labels_, return_counts=True)
        self.ref_bop = label_counts / np.sum(label_counts)
        self.prototypes = self.codebook.cluster_centers_
        self.ref_bop, _ = self.__calculate_bop(self.ref_data)
        self.__write_to_bop_file(bop_file)

    def evaluate(self, encode_dataset):

        bop, _ = self.__calculate_bop(encode_dataset)

        js = BoP.jensen_shannon_divergence(self.ref_bop, bop)
        c2 = BoP.chi2_distance(self.ref_bop, bop)
        helliger = BoP.hellinger_explicit(self.ref_bop, bop)

        results = {
                   'BoP': bop,
                   'JS': js,
                   'Hellinger' : helliger,
                   'C2' : c2
        }

        return results

    def __calculate_bop(self, samples):

        assert samples.shape[1] == self.prototypes.shape[1]
        n, d = samples.shape
        k = self.prototypes.shape[0]

        samples = samples.astype(self.codebook.cluster_centers_.dtype)
        labels = self.codebook.predict(samples)
        probs = np.zeros([k])
        label_vals, label_counts = np.unique(labels, return_counts=True)
        probs[label_vals] = label_counts / n
        return probs, labels

    def __read_from_bop_file(self, bop_file):
        if bop_file and os.path.isfile(bop_file):
            print('Loading results from', bop_file)
            bop_data = pkl.load(open(bop_file,'rb'))
            self.ref_bop = bop_data['BoP']
            self.prototypes = bop_data['Prototypes']
            self.codebook = KMeans(n_clusters=self.n_prototype, init=self.prototypes).fit(self.prototypes)
            return True
        return False

    def __write_to_bop_file(self, bop_file):
        if bop_file:
            print('Save results to', bop_file)
            bop_data = {
                        'BoP': self.ref_bop,
                        'Prototypes': self.prototypes,
                         }
            pkl.dump(bop_data, open(bop_file, 'wb'))

    @staticmethod
    def chi2_distance(A, B):

        chi = 0.5 * np.sum([((a - b) ** 2) / (a + b)
                            for (a, b) in zip(A, B)])

        return chi

    @staticmethod
    def hellinger_explicit(p, q):

        list_of_squares = []
        for p_i, q_i in zip(p, q):
            s = (np.sqrt(p_i) - np.sqrt(q_i)) ** 2

            list_of_squares.append(s)
        sosq = sum(list_of_squares)

        return sosq / np.sqrt(2)

    @staticmethod
    def jensen_shannon_divergence(p, q):
        """
        Calculates the symmetric Jensen–Shannon divergence between the two PDFs
        """
        m = (p + q) * 0.5
        return 0.5 * (BoP.kl_divergence(p, m) + BoP.kl_divergence(q, m))

    @staticmethod
    def kl_divergence(p, q):
        """
        The Kullback–Leibler divergence.
        Defined only if q != 0 whenever p != 0.
        """
        assert np.all(np.isfinite(p))
        assert np.all(np.isfinite(q))
        assert not np.any(np.logical_and(p != 0, q == 0))

        p_pos = (p > 0)
        return np.sum(p[p_pos] * np.log(p[p_pos] / q[p_pos]))
