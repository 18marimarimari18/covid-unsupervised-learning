#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

'''
No need to change this function
Can add more features if you wish, but keep it cross-sectional (i.e. one row per country)
'''
def make_data_cross_sectional(file_path):

    df = pd.read_csv("df_final.csv", index_col=0)

    # Convert the dataset from longitudinal to cross-sectional format
    cross_sectional_df = df.groupby("country").agg(
        total_weeks=("Week #", "count"),  # Count of unique weeks
        max_confirmed=("Confirmed", "max"),
        max_deaths=("Deaths", "max"),
        max_recovered=("Recovered", "max"),
        mean_confirmed=("Confirmed", "mean"),
        mean_deaths=("Deaths", "mean"),
        mean_recovered=("Recovered", "mean"),
        std_confirmed=("Confirmed", "std"),
        std_deaths=("Deaths", "std"),
        std_recovered=("Recovered", "std")
    ).reset_index()

    # Identify static columns (non-time-dependent features)
    static_columns = [
        "Health_exp_pct_GDP_2016",
        "Health_exp_public_pct_2016",
        "Health_exp_out_of_pocket_pct_2016",
        "Health_exp_per_capita_USD_2016",
        "per_capita_exp_PPP_2016",
        "Physicians_per_1000_2009-18",
        "TOTAL",  
        "ICU",
        "ACUTE"
    ]

    # Merge static columns by taking the first available value for each country
    static_data = df.groupby("country")[static_columns].first().reset_index()

    # Merge the static data with the cross-sectional dataset
    intermediate_cross_sectional_df = pd.merge(cross_sectional_df, static_data, on="country", how="left")

    # Compute new weekly cases by taking the difference between consecutive weeks
    df["weekly_confirmed"] = df.groupby("country")["Confirmed"].diff().fillna(0)
    df["weekly_deaths"] = df.groupby("country")["Deaths"].diff().fillna(0)
    df["weekly_recovered"] = df.groupby("country")["Recovered"].diff().fillna(0)

    # Aggregate to get mean and std deviation of new cases per country
    new_cases_stats = df.groupby("country").agg(
        mean_weekly_confirmed=("weekly_confirmed", "mean"),
        std_weekly_confirmed=("weekly_confirmed", "std"),
        mean_weekly_deaths=("weekly_deaths", "mean"),
        std_weekly_deaths=("weekly_deaths", "std"),
        mean_weekly_recovered=("weekly_recovered", "mean"),
        std_weekly_recovered=("weekly_recovered", "std"),
    ).reset_index()


    final_cross_sectional_df = pd.merge(intermediate_cross_sectional_df, new_cases_stats, on="country", how="left")

    final_cross_sectional_df.to_csv("cross-sect_df.csv")

    # Return the final transformed dataset
    return final_cross_sectional_df



'''
Returns: the scaled/normalized dataset, the country names, the dataset features names
TODO: Fill in the correct return values
Can choose to change how it handles missing values, or use a different Scaler if you wish
'''
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, index_col=0)
    
    print("\n--- INITIAL DATA PREVIEW ---")
    print(f"Dataset shape: {df.shape}")
    print(df.head()) # Shows the first 5 rows
    
    country_names = df['country']
    df_numeric = df.drop(columns=['country'])
    
    # Check for missing values before filling
    print("\nMissing values BEFORE fill:")
    print(df_numeric.isna().sum().sum())
    
    df_numeric.fillna(df_numeric.mean(), inplace=True)
    
    # Verify missing values are gone
    print("Missing values AFTER fill:")
    print(df_numeric.isna().sum().sum())
    
    df_cols = df_numeric.columns
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)
    
    print("\n--- SCALED DATA PREVIEW ---")
    # df_scaled is a numpy array, so we slice it to preview the first 2 rows
    print(df_scaled[:2]) 
    
    return df_scaled, country_names, df_cols


'''
Returns: nothing
TODO: Plot the elbow plot for the given data
    X-axis: Number of clusters
    Y-axis: Distortion
TODO: Save the plot as "elbow_plot.jpg"
'''
def elbow_method(data):
    distortions = []
    K_range = range(2, 11) # The range of clusters to check
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    
    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, distortions, marker='o', linestyle='--')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion (Inertia)')
    plt.title('Elbow Method For Optimal k')
    plt.savefig("elbow_plot.jpg")
    plt.show()
    


'''
Returns: nothing
TODO: Plot the silhouette plot for the given data
    X-axis: Number of clusters
    Y-axis: Score
TODO: Save the plot as "silhouette_plot.jpg"
'''
def silhouette_analysis(data):
    scores = []
    K_range = range(2, 11) # The range of clusters to check
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)
    
    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, scores, marker='o', linestyle='--', color='green')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis For Optimal k')
    plt.savefig("silhouette_plot.jpg")
    plt.show()


'''
Returns: a Dataframe of just the country names and their assigned clusters
TODO: Perform KMeans clustering on the data
Plots the k_means clustering plot
    X-axis: PCA Component 3
    Y-axis: PCA Component 4
    Color: Cluster label
TODO: Save the plot as "k_means_cluster_plot.jpg"
TODO: Fill in blanks
'''
def kmeans_clustering(data, country_names, n_clusters=3):
    # Perform the clustering and fitting
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    
    # PCA for dimensionality reduction (for plotting purposes only)
    pca = PCA(n_components=5)
    reduced_data = pca.fit_transform(data)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_data[:, 3], reduced_data[:, 4], c=labels, cmap='viridis')
    
    plt.xlabel('PCA Component 3')
    plt.ylabel('PCA Component 4')
    plt.title('K-Means Clustering')
    
    # Assigning country names to the points
    for i, country in enumerate(country_names):
        plt.annotate(country, (reduced_data[i, 3], reduced_data[i, 4]), fontsize=8, alpha=0.7)
    
    # Creating a dynamic legend based on actual clusters found
    unique_labels = np.unique(labels)
    plt.legend(handles=scatter.legend_elements()[0], labels=[f"Cluster {l}" for l in unique_labels])
    
    plt.savefig("k_means_cluster_plot.jpg")
    plt.show()
    
    # Return assigned clusters
    cluster_df = pd.DataFrame({'Country': country_names.values, 'Cluster': labels})
    return cluster_df


def kmeans_feature_importance(data, feature_names, n_clusters=3):
    # Perform the clustering and fitting
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    
    # Get cluster centers
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=feature_names)
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_centers.T, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Importance in K-Means Clustering")
    plt.xlabel("Cluster")
    plt.ylabel("Feature")
    
    # Save before showing
    plt.savefig("k_means_feature_importance_plot.jpg", bbox_inches='tight')
    plt.show()

'''
Returns: nothing
TODO: Perform Hierarchical clustering on the data
Plot the Dendogram for the given data
TODO: Save the plot as "hierarchical_cluster_plot.jpg"
TODO: Fill in blanks
'''
def hierarchical_clustering(data, country_names):
    # Perform the clustering using Ward's method
    linked = linkage(data, method='ward')
    
    # Create the plot
    plt.figure(figsize=(30, 6))
    # Pass the linked data and country names as labels
    dendrogram(linked, labels=country_names.values, leaf_rotation=90, leaf_font_size=8)
    
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Country')
    plt.ylabel('Distance')
    plt.savefig("hierarchical_cluster_plot.jpg", bbox_inches='tight')
    plt.show()


'''
Returns: a Dataframe of just the country names and their assigned clusters
TODO: Perform DBSCAN clustering on the data
Plot the DBSCAN clustering plot
    X-axis: PCA Component 3
    Y-axis: PCA Component 4
    Color: Cluster assignment
TODO: Save the plot as "dbscan_cluster_plot.jpg"
TODO: Fill in blanks
'''
def dbscan_clustering(data, country_names, eps=1.5, min_samples=5):
    # Perform clustering and fitting
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    
    # PCA
    pca = PCA(n_components=5)
    reduced_data = pca.fit_transform(data)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_data[:, 3], reduced_data[:, 4], c=labels, cmap='plasma')
    plt.xlabel('PCA Component 3')
    plt.ylabel('PCA Component 4')
    plt.title('DBSCAN Clustering')
    
    # Assigning country names to the points
    for i, country in enumerate(country_names):
        plt.annotate(country, (reduced_data[i, 3], reduced_data[i, 4]), fontsize=8, alpha=0.7)
    
    # Create legend (Handling the -1 noise cluster gracefully)
    unique_labels = np.unique(labels)
    legend_labels = [f"Cluster {l}" if l != -1 else "Noise (-1)" for l in unique_labels]
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    
    plt.savefig("dbscan_cluster_plot.jpg")
    plt.show()
    
    return pd.DataFrame({'Country': country_names.values, 'Cluster': labels})
    