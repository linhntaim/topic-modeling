import umap


class RandomFixedUMAP(umap.UMAP):
    def _fix_random_state(self, random_state):
        random_state.uniform(
            low=-10.0, high=10.0, size=(self.graph_.tocoo().shape[0], self.n_components)
        )
        return random_state

    def _fit_embed_data(self, X, n_epochs, init, random_state, **kwargs):
        random_state = self._fix_random_state(random_state)

        return super()._fit_embed_data(X, n_epochs, init, random_state, **kwargs)
