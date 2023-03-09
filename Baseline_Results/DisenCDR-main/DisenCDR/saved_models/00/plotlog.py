import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('logs.txt','\t')

plot = df.plot('epoch', 'train_loss')
plot.get_figure().savefig('Train loss.png')

plot = df.plot('epoch', 'dev_loss')
plot.get_figure().savefig('Dev loss.png')