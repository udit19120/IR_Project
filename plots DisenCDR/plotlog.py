import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('logs_sentiment.txt','\t')

plot = df.plot('epoch', 'train_loss')
plot.get_figure().savefig('Train loss sentiment.png')

plot = df.plot('epoch', 's_hit')
plot.get_figure().savefig('Source hit sentiment.png')

plot = df.plot('epoch', 's_ndcg')
plot.get_figure().savefig('Source ndcg sentiment.png')

plot = df.plot('epoch', 't_hit')
plot.get_figure().savefig('Target Hit sentiment.png')

plot = df.plot('epoch', 'dev_score')
plot.get_figure().savefig('Dev score sentiment.png')

plot = df.plot('epoch', 'best_dev_score')
plot.get_figure().savefig('Best Dev score sentiment.png')

