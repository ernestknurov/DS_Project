import numpy as np
import pandas as pd
from config import paths_to_raw_data
from os import mkdir, path


class ETL:
    def extract(self, paths: dict) -> tuple:

        """
        Extract data from a given paths.

        Parameters:
            - paths: dictionary with string path to files

        Return:
            - tuple of pandas dataframes
        """

        item_cat = pd.read_csv(paths['item_cat'])
        items = pd.read_csv(paths['items'])
        sales_train = pd.read_csv(paths["sales_train"])
        shops = pd.read_csv(paths["shops"])
        test = pd.read_csv(paths["test"])
        return item_cat, shops, items, sales_train, test

    def transform(self, dataset: tuple) -> tuple:

        """
         Transform given raw data.

         Parameters:
             - dataset: tuple of pandas dataframes

         Return:
             - tuple of transformed dataframes
        """

        item_cat, shops, items, sales_train, test = dataset

        # -Item categories
        item_cat.drop_duplicates('item_category_id', inplace=True)
        item_cat = item_cat.drop(item_cat.loc[item_cat.item_category_id < 0].index)
        item_cat.reset_index(drop=True, inplace=True)
        item_cat['item_category_name'] = item_cat['item_category_name'].str.replace("!|\?|²|\*|/", '', regex=True)

        # -Shops
        shops.drop_duplicates('shop_id', inplace=True)
        shops = shops.drop(shops.loc[shops.shop_id < 0].index)
        shops['shop_name'] = shops['shop_name'].str.replace("!|\?|²|\*|/| фран", '', regex=True)
        # --dealing with duplicates of shop_name depending on their occurrence in test set
        dup = shops.loc[shops.duplicated("shop_name", keep=False)] \
            .groupby("shop_name") \
            .agg(lambda x: list(x)) \
            .shop_id
        test_shops = test.shop_id.to_numpy()
        for idx, value in dup.items():
            value: np.ndarray = np.array(value)
            mask = np.isin(value, test_shops)
            if np.sum(mask) < 2:
                shops.loc[value, 'shop_id'] = value[mask][0] if value[mask].size > 0 else value[0]
                shops.drop_duplicates("shop_id", inplace=True)
                sales_train.loc[sales_train.shop_id.isin(value), 'shop_id'] = value[mask][0] if value[mask].size > 0 \
                    else value[0]
        shops.reset_index(drop=True, inplace=True)

        # -Items
        items.drop_duplicates(inplace=True)
        items = items.drop(items.loc[(~items.item_category_id.isin(item_cat['item_category_id'])) |
                                     (items.item_id < 0)
                                     ].index)
        items['item_name'] = items['item_name'].str.replace("!|\?|²|\*|/", '', regex=True)

        # --dealing with duplicates of item_name depending on their occurrence in test set
        dup = items.loc[items.duplicated("item_name", keep=False)] \
            .groupby("item_name") \
            .agg(lambda x: list(x)) \
            .item_id
        test_items = test.item_id.to_numpy()
        for idx, value in dup.items():
            value: np.ndarray = np.array(value)
            mask = np.isin(value, test_items)
            if np.sum(mask) < 2:
                items.loc[value, 'item_id'] = value[mask][0] if value[mask].size > 0 else value[0]
                items.drop_duplicates("item_id", inplace=True)
                sales_train.loc[sales_train.item_id.isin(value), 'item_id'] = value[mask][0] if value[mask].size > 0 \
                    else value[0]
        items.reset_index(drop=True, inplace=True)

        # -Test
        test.drop_duplicates(inplace=True)
        drop_conditions = (~test.item_id.isin(items['item_id'])) | \
                          (~test.shop_id.isin(shops['shop_id']))
        test.drop(test.loc[drop_conditions].index, inplace=True)
        test.reset_index(drop=True, inplace=True)

        # -Sales train
        sales_train['date'] = pd.to_datetime(sales_train['date'], dayfirst=True)
        sales_train.drop_duplicates(inplace=True, ignore_index=True)
        drop_conditions = (~sales_train.item_id.isin(items['item_id'])) | \
                          (~sales_train.shop_id.isin(shops['shop_id'])) | \
                          (sales_train.item_price < 0.1) | \
                          (sales_train.item_price > 178171) | \
                          (sales_train.item_cnt_day > 250) | \
                          (sales_train.item_cnt_day < -1e3)
        sales_train.drop(sales_train.loc[drop_conditions].index, inplace=True)
        sales_train.reset_index(drop=True, inplace=True)

        return item_cat, shops, items, sales_train, test

    def load(self, dataset: tuple, new_path: str = ".") -> bool:

        """
         Load transformed data into specified directory (by default current).

         Parameters:
             - dataset: tuple of pandas dataframes
             - new_path: string path to directory where data will be loaded

         Return:
             - True if function successfully completed
        """

        item_cat, shops, items, sales_train, test = dataset
        new_dir = path.join(new_path, 'cleaned_data')
        mkdir(new_dir)
        pd.to_pickle(item_cat, path.join(new_dir, "item_cat.pickle"))
        pd.to_pickle(items, path.join(new_dir, "items.pickle"))
        pd.to_pickle(sales_train, path.join(new_dir, "sales_train.pickle"))
        pd.to_pickle(shops, path.join(new_dir, "shops.pickle"))
        pd.to_pickle(test, path.join(new_dir, "test.pickle"))

        return True


if __name__ == "__main__":
    etl = ETL()
    raw_data = etl.extract(paths_to_raw_data)
    cleaned_data = etl.transform(raw_data)
    status = etl.load(cleaned_data)
    print("ETL successfully completed!") if status else print("ETL failed.")
