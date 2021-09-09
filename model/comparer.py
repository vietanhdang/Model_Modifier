from sentence_transformers import SentenceTransformer, util

model = 'paraphrase-MiniLM-L6-v2'
# model = 'paraphrase-xlm-r-multilingual-v1'


class Comparer:
    def __init__(self):
        self.model = SentenceTransformer(model)

    def compare(self, text1, text2):

        # Compute embedding for both lists
        embeddings1 = self.model.encode(text1, convert_to_tensor=True)
        embeddings2 = self.model.encode(text2, convert_to_tensor=True)

        # Compute cosine-similarities
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

        # Output the pairs with their score
        # print("{} \t\t Score: {:.4f}".format(text1, cosine_scores[0][0]))
        return cosine_scores[0][0]
