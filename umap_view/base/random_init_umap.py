import umap


class RandomInitUMAP(umap.UMAP):
    def __init__(
            self,
            n_components,
            n_neighbors,
            min_dist,
            metric,
            random_state,
    ):
        super().__init__(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            init='random',
        )
