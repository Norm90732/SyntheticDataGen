# Tests are based on this paper: https://arxiv.org/pdf/2401.10370

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance, norm, skew, kurtosis
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

class DataTesting:
    def __init__(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame, column='Close'):
        self.og_df = original_df 
        self.syn_df = synthetic_df
        self.column = column

        self.og_series = original_df[column]
        self.syn_series = synthetic_df[column]

    # Distribution tests: 
        #   Distribution and ACF plots, 
        #   Distribution distance (DY), 
        #   Earth moving distance (EMD), 
        #   *Kolmogorov-Smirnov test (KS) of sample moments, 
        #   Series distance, 
        #   *Kolmogorov-Smirnov test (KS) of returns
    
    def dist_and_acf(self, series):
        sns.histplot(series, kde=True)
        plt.title(f"Histogram & KDE")
        plt.show()

        plot_acf(series, lags=50)
        plt.title(f"ACF")
        plt.show()

    def dist_distance(self, bins=50):
        og_hist, bin_edges = np.histogram(self.og_series, bins=bins, density=True)
        syn_hist, _ = np.histogram(self.syn_series, bins=bin_edges, density=True)
        
        dy = np.sum(np.abs(og_hist - syn_hist)) * (bin_edges[1] - bin_edges[0])
        print(f"Distribution Distance (DY): {dy:.4f}")
        return dy

    def earth_moving(self):        
        emd = wasserstein_distance(self.og_series.values, self.syn_series.values)
        print(f"Earth Mover's Distance (Original vs Synthetic): {emd:.4f}")
        return emd
    
    def ks_test_moments(self, n_bootstrap=500):
        def bootstrap_moments(series):
            moments = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(series, size=len(series), replace=True)
                m = [np.mean(sample), np.std(sample), skew(sample), kurtosis(sample)]
                moments.append(m)
            return np.array(moments)
        
        og_moments = bootstrap_moments(self.og_series.values)
        syn_moments = bootstrap_moments(self.syn_series.values)
        
        ks_results = {}
        labels = ["Mean", "Std", "Skew", "Kurtosis"]
        
        for i, label in enumerate(labels):
            stat, p = ks_2samp(og_moments[:, i], syn_moments[:, i])
            ks_results[label] = (stat, p)
            print(f"KS Test of {label}: Stat={stat:.4f}, p={p:.4f}")
        
        return ks_results
    
    def ks_test_returns(self):
        og_returns = self.og_series.pct_change().dropna().values
        syn_returns = self.syn_series.pct_change().dropna().values
        
        stat, p = ks_2samp(og_returns, syn_returns)
        print(f"KS Test of Returns: Stat={stat:.4f}, p={p:.4f}")
        return stat, p


    def ks_test(self, series, title="Series"):  # need to to sample moments and returns
        # KS test against normal
        normal_sample = np.random.normal(np.mean(series), np.std(series), len(series))
        ks_stat, ks_p = ks_2samp(series, normal_sample)
        print(f"KS Test ({title}) - Stat: {ks_stat:.4f}, p-value: {ks_p:.4f}")
        return ks_stat, ks_p

    def series_distance(self, normalize=True):
        og = self.og_series.values
        syn = self.syn_series.values
        
        # Make lengths match (truncate the longer one)
        min_len = min(len(og), len(syn))
        og, syn = og[:min_len], syn[:min_len]
        
        if normalize:
            og = (og - np.mean(og)) / np.std(og)
            syn = (syn - np.mean(syn)) / np.std(syn)
        
        dist = euclidean(og, syn)
        print(f"Series Distance (Euclidean): {dist:.4f}")
        return dist


    # Correlation
        #   Inter-tenor correlation matrix,
        #   ACF score,
        #   *Fisher test of equality of correlation

    def correlation_matrix(self, df=None, title="Correlation Matrix"):
        if df is None:
            df = self.og_df
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty:
            print("No numeric columns found for correlation.")
            return None
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.title(title)
        plt.show()
        return corr_matrix

    def acf_score(self, lag_max=50):
        acf_orig = acf(self.og_series, nlags=lag_max)
        acf_syn = acf(self.syn_series, nlags=lag_max)
        score = np.linalg.norm(acf_orig - acf_syn)  # Euclidean distance
        print(f"ACF score (distance between series): {score:.4f}")
        return score

    def fisher_corr_test(self):
        df1 = self.og_df.select_dtypes(include=np.number)
        df2 = self.syn_df.select_dtypes(include=np.number)
        corr1 = df1.corr()
        corr2 = df2.corr()
        
        z1 = np.arctanh(corr1.values)
        z2 = np.arctanh(corr2.values)
        n = len(df1)
        
        z_diff = (z1 - z2) * np.sqrt(n - 3)
        p_values = 2 * (1 - norm.cdf(np.abs(z_diff)))
        
        print("Fisher correlation test p-values (sample):")
        print(pd.DataFrame(p_values, index=corr1.index, columns=corr1.columns))
        return p_values

    # Embedding
        #   t-SNE
        #   UMAP
        #   PCA
    def t_sne(self, df=None, title="t-SNE"):
        if df is None:
            df = self.original_df

        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty:
            print("No numeric columns found for t-SNE.")
            return None

        tsne = TSNE(n_components=2, random_state=42)
        coords = tsne.fit_transform(numeric_df.values)

        plt.figure(figsize=(8,6))
        plt.scatter(coords[:,0], coords[:,1], alpha=0.7)
        plt.title(title)
        plt.show()

        return coords


    def umap_test(self, df=None, title="UMAP"):
        if df is None:
            df = self.original_df

        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty:
            print("No numeric columns found for UMAP.")
            return None

        # import umap.umap_ as umap  # Correct import
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(numeric_df.values)

        plt.figure(figsize=(8,6))
        plt.scatter(coords[:,0], coords[:,1], alpha=0.7)
        plt.title(title)
        plt.show()

        return coords

    def pca_analysis(self, df=None, n_components=2, title="PCA"):
        if df is None:
            df = self.og_df
    
        # Only use numeric columns
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty:
            print("No numeric columns found for PCA.")
            return None

        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(numeric_df.values)

        plt.figure(figsize=(8,6))
        plt.scatter(coords[:,0], coords[:,1], alpha=0.7)
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()

        explained = pca.explained_variance_ratio_.cumsum()
        print(f"Cumulative variance explained: {explained[-1]:.4f}")

        return coords

    def run_tests(self):
        print("\n------ Original Series ------")
        self.dist_and_acf(self.og_series)
        self.ks_test(self.og_series, title="Original")
        self.correlation_matrix(df=self.og_df, title="Original Correlation Matrix")
        self.pca_analysis(df=self.og_df, title="Original PCA")
        self.t_sne(df=self.og_df, title="Original t-SNE")
        self.umap_test(df=self.og_df, title="Original UMAP")

        print("\n------ Synthetic Series ------")
        self.dist_and_acf(self.syn_series)
        self.ks_test(self.syn_series, title="Synthetic")
        self.correlation_matrix(df=self.syn_df, title="Synthetic Correlation Matrix")
        self.pca_analysis(df=self.syn_df, title="Synthetic PCA")
        self.t_sne(df=self.syn_df, title="Synthetic t-SNE")
        self.umap_test(df=self.syn_df, title="Synthetic UMAP")

        print("\n------ Distribution Comparison ------")
        self.dist_distance()
        self.earth_moving()
        self.series_distance()
        self.ks_test_moments()
        self.ks_test_returns()

        print("\n------ Correlation Comparison ------")
        self.acf_score()
        self.fisher_corr_test()         


    
    # Backtesting
        #   u-value histogram
        #   u-value histogram ranges
        #   u-value histogram difference from 1.0
        #   u-value breach rate (diff from theoretical) from 1.0
        #   Envelope plot
        #   *Kolmogorov-Smirnov test (KS) of u-values

    # Combination of KPIs
        #   KS of moments + KS of returns → Distribution (DIST) score
        #   Breach rates + KS of u-value → Backtest (BT) score
        #   Distribution + ACF (Fisher test) + BT → Composite score

def main():
    # Testing data using futures data
    og_df = pd.read_csv("futures_data/BZ.csv")
    syn_df = pd.read_csv("futures_data/CC.csv")

    tester = DataTesting(og_df, syn_df, column='Close')
    tester.run_tests()

if __name__ == '__main__':
    main()