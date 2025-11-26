import math

import umap


class ABFixedUMAP(umap.UMAP):
    def _fix_ab(self):
        def r(x, n=6):
            # return math.floor(x * pow(10, n)) / pow(10, n)
            return round(x, n)

        self._a = r(self._a)
        self._b = r(self._b)

    def _fit_embed_data(self, X, n_epochs, init, random_state, **kwargs):
        self._fix_ab()

        return super()._fit_embed_data(X, n_epochs, init, random_state, **kwargs)
