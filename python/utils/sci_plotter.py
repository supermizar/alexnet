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

    def bar(self, X, Y, color='lightskyblue'):
        rects = self.pl.bar(X, Y, width=0.35, facecolor=color, edgecolor='white', yerr=0.000000000001)

        def autolabel(rects):
            # attach some text labels
            for rect in rects:
                height = rect.get_height()
                self.pl.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%.4f' % height,
                        ha='center', va='bottom')
        autolabel(rects)

