import numpy as np


class Portfolio:
    def __init__(self, asset, fiat, interest_asset=0, interest_fiat=0):
        self.asset = asset
        self.fiat = fiat
        self.interest_asset = interest_asset
        self.interest_fiat = interest_fiat

    def valorisation(self, price):
        return sum([
            self.asset * price,
            self.fiat,
            - self.interest_asset * price,
            - self.interest_fiat
        ])

    def real_position(self, price):
        return (self.asset - self.interest_asset) * price / self.valorisation(price)

    def position(self, price):
        return self.asset * price / self.valorisation(price)

    def trade_to_position(self, position, price, trading_fees):
        # Repay interest
        current_position = self.position(price)
        interest_reduction_ratio = 1
        if (position <= 0 and current_position < 0):
            interest_reduction_ratio = min(1, position / current_position)
        elif (position >= 1 and current_position > 1):
            interest_reduction_ratio = min(1, (position - 1) / (current_position - 1))
        if interest_reduction_ratio < 1:
            self.asset = self.asset - (1 - interest_reduction_ratio) * self.interest_asset
            self.fiat = self.fiat - (1 - interest_reduction_ratio) * self.interest_fiat
            self.interest_asset = interest_reduction_ratio * self.interest_asset
            self.interest_fiat = interest_reduction_ratio * self.interest_fiat

        # Proceed to trade
        asset_trade = (position * self.valorisation(price) / price - self.asset)
        if asset_trade > 0:
            asset_trade = asset_trade / (1 - trading_fees + trading_fees * position)
            asset_fiat = - asset_trade * price
            self.asset = self.asset + asset_trade * (1 - trading_fees)
            self.fiat = self.fiat + asset_fiat
        else:
            asset_trade = asset_trade / (1 - trading_fees * position)
            asset_fiat = - asset_trade * price
            self.asset = self.asset + asset_trade
            self.fiat = self.fiat + asset_fiat * (1 - trading_fees)

    def update_interest(self, borrow_interest_rate):
        self.interest_asset = max(0, - self.asset) * borrow_interest_rate
        self.interest_fiat = max(0, - self.fiat) * borrow_interest_rate

    def __str__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def describe(self, price):
        print("Value : ", self.valorisation(price), "Position : ", self.position(price))

    def get_portfolio_distribution(self):
        return {
            "asset": max(0, self.asset),
            "fiat": max(0, self.fiat),
            "borrowed_asset": max(0, -self.asset),
            "borrowed_fiat": max(0, -self.fiat),
            "interest_asset": self.interest_asset,
            "interest_fiat": self.interest_fiat,
        }


class TargetPortfolio(Portfolio):
    def __init__(self, position, value, price):
        super().__init__(
            asset=position * value / price,
            fiat=(1 - position) * value,
            interest_asset=0,
            interest_fiat=0
        )


class MultiAssetPortfolio:
    def __init__(self, asset_dict, fiat, interest_rate_dict=None, interest_fiat=0.0):
        self.assets = asset_dict  # Dictionary with asset names as keys and quantities as values
        self.fiat = fiat  # Total fiat value
        self.interest_fiat = interest_fiat  # Amount of interest owed on fiat
        self.interest_rate_dict = interest_rate_dict or {asset: 0.0 for asset in
                                                         self.assets}  # Dictionary with asset names as keys and interest rates as values
        self.interest_dict = {asset: 0.0 for asset in
                              self.assets}  # Dictionary with asset names as keys and amount of interest as values

    @classmethod
    def from_random(cls, price_dict, fiat, interest_rate_dict=None, interest_fiat=0.0):
        """
        Creates a new MultiAssetPortfolio instance with randomly assigned asset quantities,
        ensuring the total cost does not exceed the available fiat.

        Parameters:
        - price_dict: A dictionary with asset names as keys and their current prices as values.
        - fiat: The total fiat value available to purchase assets.
        """
        asset_dict = {}
        remaining_fiat = fiat
        assets = list(price_dict.keys())
        np.random.shuffle(assets)  # Shuffle the list to ensure random order

        for asset in assets:
            price = price_dict[asset]
            if remaining_fiat > price:  # Only proceed if we have enough fiat to buy at least one unit
                max_quantity = remaining_fiat // price  # Maximum quantity we can afford
                quantity = np.random.randint(0, max_quantity + 1)  # Random quantity
                cost = quantity * price
                asset_dict[asset] = quantity
                remaining_fiat -= cost

        return cls(asset_dict=asset_dict, fiat=remaining_fiat, interest_rate_dict=interest_rate_dict,
                   interest_fiat=interest_fiat)

    def valorisation(self, price_dict):
        total_value = self.fiat
        for asset, quantity in self.assets.items():
            total_value += (quantity - self.interest_dict[asset]) * price_dict[asset]
        return total_value

    def real_position(self, price_dict):
        # Calculates the real position of each asset as a proportion of the total portfolio value
        positions = {}
        total_valuation = self.valorisation(price_dict)
        for asset in self.assets:
            positions[asset] = (self.assets[asset] - self.interest_dict[asset]) * price_dict[asset] / total_valuation
        return positions

    def trade_to_position(self, target_positions, price_dict, trading_fees):
        if sum(target_positions.values()) > 1:
            raise ValueError("Sum of target positions should not exceed 100%")
        valorisation = self.valorisation(price_dict)
        # FIXME: test and potentially correct asset & fiat borrowing.
        for asset, target_position in target_positions.items():
            current_position = self.real_position(price_dict)[asset]
            asset_trade = self.calculate_asset_trade(target_position, current_position, price_dict[asset], valorisation)

            self.repay_interest(asset, asset_trade)

            self.execute_trade(asset, asset_trade, price_dict[asset], trading_fees)

    def calculate_asset_trade(self, target_position, current_position, asset_price, valorisation):
        return target_position * valorisation / asset_price - current_position

    def repay_interest(self, asset, asset_trade):
        interest_reduction_ratio = self.calculate_interest_reduction_ratio(asset, asset_trade)

        if interest_reduction_ratio < 1:
            self.adjust_interest_owed(asset, interest_reduction_ratio)
            self.adjust_fiat_balance(asset, interest_reduction_ratio)

    def calculate_interest_reduction_ratio(self, asset, asset_trade):
        if self.interest_dict[asset] > 0 and asset_trade < 0:
            return min(1, -asset_trade / self.interest_dict[asset])
        elif self.interest_dict[asset] < 0 and asset_trade > 0:
            return min(1, asset_trade / -self.interest_dict[asset])
        return 1

    def adjust_interest_owed(self, asset, interest_reduction_ratio):
        interest_payment = (1 - interest_reduction_ratio) * self.interest_dict[asset] * self.interest_rate_dict[asset]
        self.fiat -= interest_payment
        self.interest_dict[asset] *= interest_reduction_ratio

    def adjust_fiat_balance(self, asset, interest_reduction_ratio):
        interest_fiat_payment = (1 - interest_reduction_ratio) * self.interest_fiat
        self.fiat -= interest_fiat_payment
        self.interest_fiat *= interest_reduction_ratio

    def execute_trade(self, asset, asset_trade, asset_price, trading_fees):
        if asset_trade > 0:
            self.buy_asset(asset, asset_trade, asset_price, trading_fees)
        else:
            self.sell_asset(asset, asset_trade, asset_price, trading_fees)

    def buy_asset(self, asset, asset_trade, asset_price, trading_fees):
        cost = asset_trade * asset_price * (1 + trading_fees)
        self.fiat -= cost
        self.assets[asset] += asset_trade

    def sell_asset(self, asset, asset_trade, asset_price, trading_fees):
        proceeds = -asset_trade * asset_price * (1 - trading_fees)
        self.fiat += proceeds
        self.assets[asset] += asset_trade  # asset_trade is negative here

    def __str__(self):
        # Create a string representation that lists all the attributes and their values
        attrs = ', '.join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"

    def describe(self, price_dict):
        # Print the total valuation and the detailed position of each asset
        print("Total Value: ", self.valorisation(price_dict))
        positions = self.real_position(price_dict)
        for asset, position in positions.items():
            print(f"Position in {asset}: {position:.2f}")

    def get_portfolio_distribution(self):
        # Creates a dictionary representing the distribution of assets, fiat, and interests
        distribution = {
            "assets": {asset: max(0, quantity) for asset, quantity in self.assets.items()},
            "fiat": max(0, self.fiat),
            "interests": {asset: self.interest_dict[asset] for asset in self.assets},
            "interest_fiat": self.interest_fiat,
        }
        # Include borrowed assets if any
        distribution["borrowed_assets"] = {asset: max(0, -quantity) for asset, quantity in self.assets.items() if
                                           quantity < 0}
        distribution["borrowed_fiat"] = max(0, -self.fiat) if self.fiat < 0 else 0
        return distribution
if __name__ == "__main__":
    # Example instantiation
    assets = {'AAPL': 10, 'GOOG': 5}
    fiat = 1000
    interest_rates = {'AAPL': 0.01, 'GOOG': 0.02}
    portfolio = MultiAssetPortfolio(assets, fiat, interest_rates)

    # Example valuation
    current_prices = {'AAPL': 150, 'GOOG': 2500}
    print(portfolio.valorisation(current_prices))
