from collections import defaultdict
import random
import numpy as np
import pandas as pd


class SimulDataset:
    def __init__(self, num_queries=100, embedding_dim=100, total_num_docs=1000):
        self.num_queries = num_queries
        self.embedding_dim = embedding_dim
        self.total_num_docs = total_num_docs
        self.documents = SimulDataset.setup_documents(
            embedding_dim, total_num_docs, normalize=True
        )

        self.queries = {}
        self.queries_to_doc = {}
        for split in ["train", "valid", "test"]:
            print(f"Setting up {split} queries")
            (
                self.queries[split],
                self.queries_to_doc[split],
            ) = SimulDataset.setup_queries(
                embedding_dim, num_queries, total_num_docs, normalize=True
            )

    # this is a private but not a static method
    @staticmethod
    def setup_documents(embedding_dim, total_num_docs, normalize=True):
        print("Setting up documents")
        documents = {}
        for doc_id in range(total_num_docs):
            documents[doc_id] = np.random.rand(embedding_dim)

        # normalize the document vectors
        if normalize:
            for doc_id in documents:
                documents[doc_id] /= np.linalg.norm(documents[doc_id])

        return documents

    @staticmethod
    def setup_queries(embedding_dim, num_queries, total_num_docs, normalize=True):
        qids = np.arange(num_queries)

        queries = {}
        query_to_doc_ids = {}

        # setup query vectors and their document mappings
        for qid in qids:
            queries[qid] = np.random.rand(embedding_dim)

            # setup document mappings for this query
            # by randomly sampling some document ids
            # from the total number of documents
            num_docs_per_this_query = random.randint(1, 100)
            query_to_doc_ids[qid] = np.random.choice(
                total_num_docs, num_docs_per_this_query, replace=False
            )

        # normalize the document vectors
        if normalize:
            for q_id in qids:
                queries[q_id] /= np.linalg.norm(queries[q_id])

        return queries, query_to_doc_ids

    def get_dataset(self, split):
        qids = list(self.queries[split].keys())
        query_to_doc_ids = self.queries_to_doc[split]
        queries = self.queries[split]

        # query_doc_labels is a dictionary of document labels for each query id
        query_doc_labels = defaultdict(list)

        for qid in qids:
            for doc_id in query_to_doc_ids[qid]:
                if (
                    np.dot(
                        queries[qid],
                        self.documents[doc_id],
                    )
                    > 0.75
                ):
                    query_doc_labels[qid].append(1)
                else:
                    query_doc_labels[qid].append(0)

        # create a list of tuples from the dictionaries
        data = []
        for qid in qids:
            for docid, qdoclabel in zip(query_to_doc_ids[qid], query_doc_labels[qid]):
                qvector = queries[qid]
                data.append((qid, docid, qdoclabel, qvector))

        # convert data to Pandas DataFrame
        df = pd.DataFrame(
            data=data, columns=["q_id", "doc_id", "q_doc_label", "q_vector"]
        )

        return df
