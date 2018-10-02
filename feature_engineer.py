"""
This module defines a class that assists machine learning feature engineering.

Author: Jenny Zhang, Justin Duan
Time: 2018/09/10 05:08PM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureEngineer(object):
    """
    This class provides a few class methods that assists machine learning feature engineering.
    """
    @classmethod
    def get_missing_summary(cls, df, features, target):
        """
        get a summary of missing values for the specified columns.
        :param df: input pd.DataFrame
        :param features: feature column names
        :param target: target (prediction) column name
        :return: a summary pd.Series
        """
        return df[cls._to_list(features) + cls._to_list(target)].isnull().sum()

    @classmethod
    def check_dependence_on_missing(cls, df, features, target, aggfunc=[len, np.median, np.mean, np.std]):
        """
        Check the dependence of the requested statistics of the target column on the missing pattern of the requested
        features.
        :param df: input pd.DataFrame
        :param features: feature column names
        :param target: target (prediction) column name
        :param aggfunc: aggregation function passed to pd.pivot_table
        :return: a summary pd.Series
        """
        features = cls._to_list(features)
        dfs = []
        for feature in features:
            # Continue if df[feature] contains no missing data
            if df[feature].isnull().sum() == 0:
                continue
            tmp_df = pd.pivot_table(df, index=df[feature].isnull(), values=target, aggfunc=aggfunc)
            tmp_df.index = [[feature] * len(tmp_df), tmp_df.index.values]
            dfs.append(tmp_df)
        if len(dfs) == 0:
            print("You are lucky! No missing data found!")
            return None
        return pd.concat(dfs)

    @classmethod
    def check_dependence_on_missing_hist(cls, df, features, target, bins=20, combined_view=True,
                                         percentile_range=(0.1, 99.9)):
        """
        Visualize the dependence of the requested statistics of the target column on the missing pattern of the
        requested features.
        :param df: input pd.DataFrame
        :param features: feature column names
        :param target: target (prediction) column name
        :param bins: number of bins for the histogram plot
        :param combined_view: whether or not the missing and non-missing histograms should be plotted together
        :param percentile_range: plot range in terms of data percentiles
        :return: None
        """
        features = cls._to_list(features)
        if not hasattr(percentile_range, '__iter__') and len(percentile_range) != 2:
            raise ValueError("The percentile range should be a tuple or list of two values.")

        plot_range = [df[target].quantile(x / 100) for x in percentile_range]
        for feature in features:
            feature_null = df[feature].isnull()
            if feature_null.sum() == 0:
                continue
            groups = df[target].groupby(feature_null)
            if combined_view:
                f, ax = plt.subplots()
            else:
                f, axarr = plt.subplots(ncols=len(groups))
            for i, (name, group) in enumerate(groups):
                if combined_view:
                    group.plot.hist(bins=bins, range=plot_range, label='{} missing = {}'.format(feature, name))
                    ax = plt.gca()
                    ax.set_xlim(plot_range)
                    ax.set_xlabel(target)
                    ax.legend()
                else:
                    group.plot.hist(bins=bins, ax=axarr[i], range=plot_range, label='{} missing = {}'.format(feature,
                                                                                                             name))
                    axarr[i].set_xlim(plot_range)
                    axarr[i].set_xlabel(target)
                    axarr[i].legend()
            plt.show()

    @classmethod
    def pearson_coeffs(cls, df, features):
        """
        Compute pairwise pearson coefficients for the requested features.
        :param df: input pd.DataFrame
        :param features: feature column names
        :return: a pd.DataFrame with pearson coefficients
        """
        features = cls._to_list(features)
        return df[features].corr()

    @classmethod
    def search_colinear_features(cls, df, coeff_matrix, coeff_range=(0.8, 1), plot=False, marker_size=1):
        """
        Find and visualize colinear features.
        :param df: input pd.DataFrame
        :param coeff_matrix: pearson coefficient matrix
        :param coeff_range: pairwise pearson coefficient range
        :param plot: show the bivariate plots or not
        :param marker_size: default marker size
        :return: a dictionary of found feature pairs that matches the requirements
        """
        if not hasattr(coeff_range, '__iter__') and len(coeff_range) != 2:
            raise ValueError("The percentile range should be a tuple or list of two values.")

        rlt = {}
        for i, feature in enumerate(coeff_matrix):
            for j in range(i + 1, len(coeff_matrix)):
                if coeff_range[0] <= abs(coeff_matrix.iat[j, i]) <= coeff_range[1]:
                    rlt[(feature, coeff_matrix.index[j])] = coeff_matrix.iloc[j, i]
                    if plot:
                        x, y = feature, coeff_matrix.index[j]
                        f, ax = plt.subplots()
                        df.plot.scatter(x, y, ax=ax, s=marker_size)
                        plt.xlabel(x)
                        plt.ylabel(y)
        
                    plt.show()
        return rlt

    @classmethod
    def hist(cls, df, features, gaussian_fit=True, bins=20, percentile_range=(0.1, 99.9), figsize=(5, 5)):
        """
        Visualize the distribution of each feature.
        :param df: input pd.DataFrame
        :param features: feature column names
        :param gaussian_fit: apply gaussian fit
        :param bins: number of bins for the histogram plot
        :param percentile_range: plot range in terms of data percentiles
        :param figsize: matplotlib figure size
        :return: None
        """
        features = cls._to_list(features)
        if not hasattr(percentile_range, '__iter__') and len(percentile_range) != 2:
            raise ValueError("The percentile range should be a tuple or list of two values.")
        
        # A nested Gaussian function for fit
        def _gassian(x, ampl, mu, sigma):
            return ampl * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

        # Computer the number of subplots
        if len(features) == 0:
            _, axes = plt.subplots(1, 1, figsize=(5, 5))
            axarr = [axes]
        else:
            n_cols = np.uint(np.ceil(np.sqrt(len(features))))
            n_rows = len(features) // n_cols if len(features) % n_cols == 0 else len(features) // n_cols + 1
            _, axes = plt.subplots(n_cols, n_rows, figsize=(5, 5))
            axarr = np.ravel(axes)

        # Adjust the white space between subplots
        plt.subplots_adjust(hspace=0.4)

        # Plot
        for ax, feature in zip(axarr, features):
            x = df[feature]
            x = x[~np.isnan(x)]
            ax.hist(x, bins=bins)
            ax.set_title(feature)

            # Fit to gaussian
            try:
                if gaussian_fit:
                    hist, bin_edges = np.histogram(x, bins=bins, density=False)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    coeff, var_matrix = curve_fit(_gassian, bin_centers, hist, p0=[1, np.median(x), np.std(x)])
                    hist_fit = _gassian(bin_centers, *coeff)
                    ax.plot(bin_centers, hist_fit)
            except RuntimeError:
                print("Gaussian fit for feature {} failed.".format(feature))
                pass

            # Set the plot range
            plot_range = [np.percentile(x, q) for q in percentile_range]
            ax.set_xlim(plot_range)
        plt.show()

    @classmethod
    def standardize(cls, df, features, suffix="_st", inplace=False, with_mean=True, with_std=True):
        """
        Remove the mean and scale the standard deviation to unit.
        :param df: input pd.DataFrame
        :param features: feature column names
        :param suffix: column name suffix
        :param inplace: if True, try to avoid a copy and do inplace scaling instead
        :param with_mean: If True, center the data before scaling
        :param with_std: If True, scale the data to unit variance
        :return: the scalar object and a pd.DataFrame with standardized new features if inplace=False, otherwise return
                 the scalar object and the input DataFrame
        """
        features = cls._to_list(features)
        scalar = StandardScaler(copy=True, with_mean=with_mean, with_std=with_std)
        transformed_data = scalar.fit_transform(df[features])
        if inplace:
            df[features] = pd.DataFrame(transformed_data, columns=features, index=df.index)
            return scalar, df
        else:
            new_features = ["{}{}".format(x, suffix) for x in features]
            return scalar, pd.DataFrame(transformed_data, columns=new_features, index=df.index)

    @classmethod
    def pca(cls, df, features, n_components=None, prefix="prin_"):
        """
        Apply principle component analysis for the requested features.
        :param df: input pd.DataFrame
        :param features: feature column names
        :param n_components: number of components to return. If None, n_components = len(features)
        :param prefix: new feature name prefix
        :return: the pca object and a pd.DataFrame with the new principle components
        """
        features = cls._to_list(features)
        n_components = len(features) if n_components is None else n_components
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(df[features])
        new_features = ['{}{}'.format(prefix, i) for i in range(n_components)]
        df_pca = pd.DataFrame(data=transformed_data, columns=new_features, index=df.index)
        return pca, df_pca

    @classmethod
    def _to_list(cls, obj):
        """Wrap an object inside a list if it's not one already."""
        return obj if hasattr(obj, '__iter__') and not isinstance(obj, str) else [obj]


if __name__ == '__main__':
    pass