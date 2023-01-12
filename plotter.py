import pandas as pd

def plot():
    df = pd.read_csv(filepath_or_buffer='results.csv', header=None, names=['Потоков на блок', 'Время'])
    print(df.head())
    fig = df.plot(kind='line', title='4096x2048 * 2048x8192', x='Потоков на блок', y='Время', grid=True, logy=True).get_figure()
    fig.savefig('plot.png')


plot()
