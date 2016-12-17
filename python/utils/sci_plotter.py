import pylab as pl


class SciPlot(object):
    def __init__(self, title=''):
        self.pl = pl
        self.pl.title(title)

    def plot(self, arr, desc=''):
        self.pl.plot(arr, label=desc)
        self.pl.legend()

    def show(self):
        self.pl.show()

    def bar(self, x, y):
        self.pl.bar(x, y, align="center", width=0.5, alpha=0.5)
        self.pl.show()
