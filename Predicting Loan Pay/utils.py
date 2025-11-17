from scipy import stats
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class Plot:

    @staticmethod
    def hist(data, ax, title=''):
        ax.hist(data, bins='auto', )
        ax.set_xlabel(title, fontsize=20, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=20, fontweight='bold')

    @staticmethod
    def boxplot(data, ax, labels=[], title=''):
        ax.boxplot(data, labels=labels)
        ax.set_title(title)

    @staticmethod
    def lineplot(x, y, ax, title=''):
        ax.plot(x,y)
        ax.set_title(title, fontsize=20, fontweight='bold')

class Static:

    @staticmethod
    def Ttest(group0, group1):

        # Test for homogeneity of variance
        levene_stat, levene_p = stats.levene(group0, group1)
        # set the parameter of equal_var according to the result of test for homogeneity of variance
        equal_var = True if levene_p > 0.05 else False
        # Sample independent T-test
        t_statistic, p_value = stats.ttest_ind(group0, group1, equal_var=equal_var)

        return p_value

    @staticmethod
    def PBCtest(continuous, label):
        '''
        point biserial correlation test, the formula is "(X0-X1)*(pq)^0.5/Var",
         where x0 is the average of x of label0 and x0 is of label1, and p、q is the ratio of 0 and 1 in turn
         Var is the variance of x in full data.
        :param continuous: continue data
        :param label: label 0 or 1
        '''
        correlation, p_value = stats.pointbiserialr(label, continuous)
        return correlation

    @staticmethod
    def Chi2test(category, label):
        '''
        use contingency table and chi2 test to identify the correlation between category and label
        :param category: category data
        :param label: label 0 or 1
        '''
        # Generate contingency table
        contingency_table = pd.crosstab(category, label)
        # Chi2 test
        chi2, p_value, dof, expected_freq = stats.chi2_contingency(contingency_table)
        # Calculate the Cramer's V
        n = contingency_table.sum().sum()  # amount of samples
        min_dim = min(contingency_table.shape) - 1  # degree of freedom
        cramers_v = np.sqrt(chi2 / (n * min_dim))

        return cramers_v, p_value

    @staticmethod
    def numpy_groupby(key_data, value_data, func):
        '''
        :param key_data: groupby column
        :param value_data: value column
        :param func: aggregate function
        '''
        # 1. Sort the values according to grouped keys
        sort_indices = np.argsort(key_data)
        sorted_keys = key_data[sort_indices]
        sorted_values = value_data[sort_indices]
        # 2. Find the change points when the key changes(thus the beginning index of a new group)
        diff_indices = np.where(sorted_keys[:-1] != sorted_keys[1:])[0]+1
        grouped_arrays = np.split(sorted_values, diff_indices)
        # 3. Attain the unique keys and apply agg function
        unique_keys = np.unique(sorted_keys)
        aggregated_values = [func(arr) for arr in grouped_arrays]

        return dict(zip(unique_keys, aggregated_values))

    @staticmethod
    def IV(category, label):
        '''
        calculate IV of category data on its label 0/1
        :param category:
        :param label:
        :return:
        '''
        label1 = np.sum(label)+0.5
        label0 = len(label)-label1+1

        # Define agg function
        def agg_func(arr):
            cnt1 = np.sum(arr)+0.5
            cnt0 = len(arr)-cnt1+1
            return (cnt1/label1 - cnt0/label0)*np.log(cnt1*label0/(cnt0*label1))

        grouped_iv = Static.numpy_groupby(category, label, agg_func)
        sums = 0
        for v in grouped_iv.values():
            sums += v
        return sums/len(grouped_iv)

class SOSEncoder(BaseEstimator, TransformerMixin):
    """
    SOS Encoder that splits continuous features to sparse features,
    internal cross-validation for leakage prevention, and smoothing.

    Parameters
    ----------
    bins : str, default=5
        Number of bins that cut into.

    cv : int, default=5
        Number of folds for cross-validation in fit_transform.
    """

    def __init__(self, bins=10, cv=5):
        self.bins = bins
        self.zoom_ = np.power(2, bins)
        self.cv = cv
        self.q95_ = None
        self.q5_ = None
        self.columns = None
        self.res_columns = []

    def fit(self, X):
        try:
            self.columns = X.columns.copy()
        except:
            self.columns = [f'{i}' for i in range(X.shape[1])]
        for col in self.columns:
            for i in range(self.bins+2):
                self.res_columns.append(f'{col}_sos{i}')

        self.q95_ = np.quantile(X, 0.95, axis=0)
        self.q5_ = np.quantile(X, 0.05, axis=0)
        # return to the encoder itself to support chain calls (pipeline)
        return self

    def transform(self, X):
        # First, trunc the data to exclude the abnormal data
        X_array = np.clip(X, self.q5_, self.q95_)
        # Secondary, zoom data and take log, ensure enough bins can be filled
        X_array = np.log2((X_array-self.q5_)/(self.q95_-self.q5_)*self.zoom_+1)
        # Third, take floor and ceil
        floor = np.floor(X_array)
        ceil = np.ceil(X_array)
        # Forth, calculate the weights of floor and ceil
        floor_weight = X_array - floor
        ceil_weight = ceil - X_array
        # Fifth, use sos encoding
        res = pd.DataFrame(columns=self.res_columns, index=X_array.index, dtype=np.float64)
        for i in range(X_array.shape[1]):
            ## floor ##
            sorted_indices = np.argsort(floor.iloc[:, i])
            sorted_keys = floor.iloc[sorted_indices, i]
            unique_keys = np.unique(sorted_keys) # unique bins' labels
            diff_indices = np.where(sorted_keys[:-1].values != sorted_keys[1:].values)[0] + 1 # split node
            split_indices = np.split(sorted_indices, diff_indices) # split the sorted index
            for j in range(len(unique_keys)):
                col = f'{self.columns[i]}_sos{int(unique_keys[j])}'
                res.loc[X.index[split_indices[j]], col] = floor_weight.iloc[split_indices[j], i].copy()
            ## ceil ##
            sorted_indices = np.argsort(ceil.iloc[:,i])
            sorted_keys = ceil.iloc[sorted_indices, i]
            unique_keys = np.unique(sorted_keys)
            diff_indices = np.where(sorted_keys[:-1].values != sorted_keys[1:].values)[0] + 1
            split_indices = np.split(sorted_indices, diff_indices)
            for j in range(len(unique_keys)):
                col = f'{self.columns[i]}_sos{int(unique_keys[j])}'
                res.loc[X.index[split_indices[j]], col] = ceil_weight.iloc[split_indices[j], i].copy()

        return res

    def fit_transform(self , X):
        """
        Fit and transform the data using internal cross-validation to prevent leakage.
        """
        # First, fit on the entire dataset to get 95/5 quantile
        self.fit(X)
        # Secondly, use cv
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        res = pd.DataFrame(columns=self.res_columns, index=X.index, dtype=np.float64)
        for train_idx, val_idx in kf.split(X):
            train= X.iloc[train_idx]
            val = X.iloc[val_idx]
            # attain the top and bottom using training data
            q95 = np.quantile(train, 0.95, axis=0)
            q5 = np.quantile(train, 0.05, axis=0)
            val = np.clip(val, q5, q95)
            # logarithm
            val = np.log2((val - q5) / (q95 - q5) * self.zoom_ + 1)
            # floor and ceil
            floor = np.floor(val)
            ceil = np.ceil(val)
            # weights of floor and ceil
            floor_weight = val - floor
            ceil_weight = ceil - val
            # sos encoding
            for i in range(len(self.columns)):
                ## floor ##
                sorted_indices = np.argsort(floor.iloc[:, i])
                sorted_keys = floor.iloc[sorted_indices, i]
                unique_keys = np.unique(sorted_keys)
                diff_indices = np.where(sorted_keys[:-1].values != sorted_keys[1:].values)[0] + 1
                split_indices = np.split(sorted_indices, diff_indices)
                for j in range(len(unique_keys)):
                    col = f'{self.columns[i]}_sos{int(unique_keys[j])}'
                    res.loc[val.index[split_indices[j]], col] = floor_weight.iloc[split_indices[j], i].copy()
                ## ceil ##
                sorted_indices = np.argsort(ceil.iloc[:, i])
                sorted_keys = ceil.iloc[sorted_indices, i]
                unique_keys = np.unique(sorted_keys)
                diff_indices = np.where(sorted_keys[:-1].values != sorted_keys[1:].values)[0] + 1
                split_indices = np.split(sorted_indices, diff_indices)
                for j in range(len(unique_keys)):
                    col = f'{self.columns[i]}_sos{int(unique_keys[j])}'
                    res.loc[val.index[split_indices[j]], col] = ceil_weight.iloc[split_indices[j], i].copy()

        return res

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder that supports multiple aggregation functions,
    internal cross-validation for leakage prevention, and smoothing.

    Parameters
    ----------
    cols_to_encode : list of str
        List of column names to be target encoded.

    aggs : list of str, default=['mean']
        List of aggregation functions to apply. Any function accepted by
        pandas' `.agg()` method is supported, such as:
        'mean', 'std', 'var', 'min', 'max', 'skew', 'nunique',
        'count', 'sum', 'median'.
        Smoothing is applied only to the 'mean' aggregation.

    cv : int, default=5
        Number of folds for cross-validation in fit_transform.

    smooth : float or 'auto', default='auto'
        The smoothing parameter `m`. A larger value puts more weight on the
        global mean. If 'auto', an empirical Bayes estimate is used.

    drop_original : bool, default=False
        If True, the original columns to be encoded are dropped.
    """

    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        """
        Learn mappings from the entire dataset.
        These mappings are used for the transform method on validation/test data.
        """
        temp_df = X.copy()
        temp_df['target'] = y

        # Learn global statistics for each aggregation
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)

        # Learn category-specific mappings
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping

        return self

    def transform(self, X):
        """
        Apply learned mappings to the data.
        Unseen categories are filled with global statistics.
        """
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series)
                X_transformed[new_col_name].fillna(self.global_stats_[agg_func], inplace=True)

        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)

        return X_transformed

    def fit_transform(self, X, y):
        """
        Fit and transform the data using internal cross-validation to prevent leakage.
        """
        # First, fit on the entire dataset to get global mappings for transform method
        self.fit(X, y)

        # Initialize an empty DataFrame to store encoded features
        encoded_features = pd.DataFrame(index=X.index)

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                # --- Calculate mappings only on the training part of the fold ---
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'

                    # Calculate global stat for this fold
                    fold_global_stat = y_train.agg(agg_func)

                    # Calculate category stats for this fold
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)

                    # --- Apply smoothing only for 'mean' aggregation ---
                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()

                        m = self.smooth
                        if self.smooth == 'auto':
                            # Empirical Bayes smoothing
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if variance_between > 0:
                                m = avg_variance_within / variance_between
                            else:
                                m = 0  # No smoothing if no variance between groups

                        # Apply smoothing formula
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)

                    # Store encoded values for the validation fold
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)

        # Merge with original DataFrame
        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]

        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)

        return X_transformed

class EqualSplit:
    """
    Equal Splitter that splits the data according to the label and ensure the label ratio is balanced in each subsample data

    Parameters
    ----------
    random_state : int, default=42
        The random seed.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit_resample(self, X, y):
        X_array = X.copy()
        y_array = y.copy()

        # Shuffle X and y synchronously
        np.random.seed(self.random_state)
        shuffled_indices = np.random.permutation(X_array.index)
        X_array = X_array.reindex(shuffled_indices)
        y_array = y_array.reindex(shuffled_indices)


        uniques = y_array.unique()
        if len(uniques) != 2:
            raise NotImplementedError('the n unique of target must be 2.')

        # Analyse the majority and minority label, calculate the split ratio and split node
        cnt_majority = np.sum(y_array == uniques[0])
        cnt_minority = X.shape[0] - cnt_majority
        if cnt_majority >= cnt_minority:
            majority, minority = uniques
        else:
            minority, majority = uniques
            cnt_majority, cnt_minority = cnt_minority, cnt_majority
        n_split = round(cnt_majority / cnt_minority) # number of the batch to be split into
        split_node = cnt_majority // n_split # index to be split

        self.n_split_ = n_split
        self.majority_ = split_node
        self.minority_ = cnt_minority

        # Minority and minority index
        minor_idx = y_array[y_array == minority].index
        major_idx = y_array[y_array == majority].index

        # Generate the equal split data
        right = 0
        res_X = [] # contain the split data X
        res_y = [] # contain the split data y
        for i in range(n_split):
            left = right
            if i == n_split - 1:
                right = cnt_majority
            else:
                right = right + split_node
            idx = minor_idx.append(major_idx[left:right])
            idx = np.random.permutation(idx)
            res_X.append(X_array.loc[idx])
            res_y.append(y_array.loc[idx])

        return res_X, res_y
