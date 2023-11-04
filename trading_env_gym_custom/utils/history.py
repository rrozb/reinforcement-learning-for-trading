import numpy as np


class History:
    def __init__(self, max_size=10000):
        self.height = max_size

    def set(self, **kwargs):
        # Flattening the inputs to put it in np.array
        self.columns = []
        for name, value in kwargs.items():
            if isinstance(value, list):
                self.columns.extend([f"{name}_{i}" for i in range(len(value))])
            elif isinstance(value, dict):
                self.columns.extend([f"{name}_{key}" for key in value.keys()])
            else:
                self.columns.append(name)

        self.width = len(self.columns)
        self.history_storage = np.zeros(shape=(self.height, self.width), dtype='O')
        self.size = 0
        self.add(**kwargs)

    def add(self, **kwargs):
        values = []
        columns = []
        for name, value in kwargs.items():
            if isinstance(value, list):
                columns.extend([f"{name}_{i}" for i in range(len(value))])
                values.extend(value[:])
            elif isinstance(value, dict):
                columns.extend([f"{name}_{key}" for key in value.keys()])
                values.extend(list(value.values()))
            else:
                columns.append(name)
                values.append(value)

        if columns == self.columns:
            self.history_storage[self.size, :] = values
            self.size = min(self.size + 1, self.height)
        else:
            raise ValueError(
                f"Make sur that your inputs match the initial ones... Initial ones : {self.columns}. New ones {columns}")

    def __len__(self):
        return self.size

    def __getitem__(self, arg):
        if isinstance(arg, tuple):
            column, t = arg
            try:
                column_index = self.columns.index(column)
            except ValueError as e:
                raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
            return self.history_storage[:self.size][t, column_index]
        if isinstance(arg, int):
            t = arg
            return dict(zip(self.columns, self.history_storage[:self.size][t]))
        if isinstance(arg, str):
            column = arg
            try:
                column_index = self.columns.index(column)
            except ValueError as e:
                raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
            return self.history_storage[:self.size][:, column_index]
        if isinstance(arg, list):
            columns = arg
            column_indexes = []
            for column in columns:
                try:
                    column_indexes.append(self.columns.index(column))
                except ValueError as e:
                    raise ValueError(
                        f"Feature {column} does not exist ... Check the available features : {self.columns}")
            return self.history_storage[:self.size][:, column_indexes]

    def __setitem__(self, arg, value):
        column, t = arg
        try:
            column_index = self.columns.index(column)
        except ValueError as e:
            raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
        self.history_storage[:self.size][t, column_index] = value


class MultiAssetHistory:
    def __init__(self, assets, max_size=10000):
        # Initialize a History instance for each asset
        self.histories = {asset: History(max_size=max_size) for asset in assets}

    #FIXME: refactor to not have duplicate data for each asset. Like idx, reward etc.

    def _add_asset(self, asset, **kwargs):
        # The original 'add' method now private
        self.histories[asset].add(**kwargs)

    def _set_asset(self, asset, **kwargs):
        # The original 'set' method now private
        self.histories[asset].set(**kwargs)

    def add(self, **kwargs):
        # New 'add' method that ensures atomicity across all assets
        # Perform validation here if necessary
        for asset in self.histories.keys():
            if asset not in kwargs:
                raise ValueError(f"Missing data for asset: {asset}")
            self._add_asset(asset, **kwargs[asset])

    def set(self, **kwargs):
        # New 'set' method that ensures atomicity across all assets
        # Perform validation here if necessary
        for asset in self.histories.keys():
            if asset not in kwargs:
                raise ValueError(f"Missing data for asset: {asset}")
            self._set_asset(asset, **kwargs[asset])