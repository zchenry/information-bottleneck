from lib import *

def pr(s, l=100):
    print(' ' * l, end='\r')
    print(s, end='\r')

def tuple2array(data, xp):
    xs = xp.array([d[0] for d in data], dtype=xp.float32)
    ys = xp.array([d[1] for d in data])
    return xs, ys

class MLP(chainer.Chain):
    def __init__(self, hs, out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            self.ls = chainer.ChainList(
                *[L.Linear(None, h) for h in hs + [out]])

    def forward(self, x, keep_val=False):
        if keep_val:
            vals = []

        for l in self.ls[:-1]:
            x = F.arctan(l(x))
            if keep_val:
                vals.append(x)

        output = self.ls[-1](x)
        if keep_val:
            vals.append(output)

        if keep_val:
            return output, vals
        else:
            return output

    def loss(self, x, y):
        return F.softmax_cross_entropy(self.forward(x), y)
