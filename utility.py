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
            self.hs = hs + [10]
            for i, h in enumerate(self.hs):
                n_in = 784 if i == 0 else hs[i-1]
                setattr(self, 'w{}'.format(i),
                        chainer.Parameter(
                            initializers.Normal(scale=np.sqrt(1./n_in)), (h, n_in)))
                setattr(self, 'b{}'.format(i),
                        chainer.Parameter(initializers.Zero(), (h,)))

    def forward(self, x, keep_val=False):
        if keep_val:
            vals = []

        for i in range(len(self.hs) - 1):
            w = getattr(self, 'w{}'.format(i))
            b = getattr(self, 'b{}'.format(i))
            x = F.arctan(F.linear(x, w, b))
            if keep_val:
                vals.append(x)

        w = getattr(self, 'w{}'.format(len(self.hs) - 1))
        b = getattr(self, 'b{}'.format(len(self.hs) - 1))
        output = F.linear(x, w, b)
        if keep_val:
            vals.append(output)

        if keep_val:
            return output, vals
        else:
            return output

    def loss(self, x, y):
        return F.softmax_cross_entropy(self.forward(x), y)
