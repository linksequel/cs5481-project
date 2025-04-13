import random
import math
import pandas as pd


class BiasSVD():
    def __init__(self, rating_data, F=5, alpha=0.1, lmbda=0.1, max_iter=100):
        self.F = F
        self.P = dict()
        self.Q = dict()
        self.bu = dict()
        self.bi = dict()
        self.mu = 0
        self.alpha = alpha
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.rating_data = rating_data

        users = rating_data['user'].unique()
        items = rating_data['item'].unique()

        for user in users:
            self.P[user] = [random.random() / math.sqrt(self.F) for x in range(0, F)]
            self.bu[user] = 0
        for item in items:
            self.Q[item] = [random.random() / math.sqrt(self.F) for x in range(0, F)]
            self.bi[item] = 0

    def train(self):
        mu_sum = self.rating_data['rating'].sum()
        cnt = len(self.rating_data)
        self.mu = mu_sum / cnt

        for step in range(self.max_iter):
            for index, row in self.rating_data.iterrows():
                user = row['user']
                item = row['item']
                rui = row['rating']
                rhat_ui = self.predict(user, item)
                e_ui = rui - rhat_ui

                self.bu[user] += self.alpha * (e_ui - self.lmbda * self.bu[user])
                self.bi[item] += self.alpha * (e_ui - self.lmbda * self.bi[item])
                for k in range(0, self.F):
                    self.P[user][k] += self.alpha * (e_ui * self.Q[item][k] - self.lmbda * self.P[user][k])
                    self.Q[item][k] += self.alpha * (e_ui * self.P[user][k] - self.lmbda * self.Q[item][k])
            self.alpha *= 0.1

    def predict(self, user, item):
        return sum(self.P[user][f] * self.Q[item][f] for f in range(0, self.F)) + self.bu[user] + self.bi[
            item] + self.mu


def loadData():
    data = {
        'user': [1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5],
        'item': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E'],
        'rating': [5, 3, 4, 4, 3, 1, 2, 3, 3, 4, 3, 4, 3, 5, 3, 3, 1, 5, 4, 1, 5, 5, 2, 1]
    }
    return pd.DataFrame(data)


rating_data = loadData()
basicsvd = BiasSVD(rating_data, F=10)
basicsvd.train()
for item in ['E']:
    print(item, basicsvd.predict(1, item))