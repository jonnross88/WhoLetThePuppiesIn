import unittest
import pandas as pd
import numpy as np
import helper_functions as hf
from sklearn.cluster import KMeans
from sklearn import metrics


class TestCalculateClustersScores(unittest.TestCase):
    def setUp(self):
        # Create a sample dataframe for testing
        self.data_dict = {"PCA": pd.DataFrame(np.random.rand(100, 5))}
        self.nclusters = list(range(2, 5))

    def test_calculate_clusters_scores(self):
        # Call the function with the test data
        result = hf.calculate_clusters_scores(self.data_dict, self.nclusters)

        # Check that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check that the DataFrame has the expected columns
        expected_columns = [
            "embedding_key",
            "n_clusters",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
        ]
        self.assertTrue(all([col in result.columns for col in expected_columns]))

    def test_calculate_clusters_scores_logic(self):
        # Call the function with the test data
        result = hf.calculate_clusters_scores(self.data_dict, self.nclusters)

        # For each row in the result, check that the silhouette, calinski_harabasz, and davies_bouldin scores are correct
        for _, row in result.iterrows():
            # Get the actual scores
            actual_silhouette = row["silhouette"]
            actual_calinski_harabasz = row["calinski_harabasz"]
            actual_davies_bouldin = row["davies_bouldin"]

            # Calculate the expected scores
            embedding = self.data_dict[row["embedding_key"]]
            labels = KMeans(n_clusters=row["n_clusters"]).fit_predict(embedding)
            expected_silhouette = metrics.silhouette_score(embedding, labels)
            expected_calinski_harabasz = metrics.calinski_harabasz_score(
                embedding, labels
            )
            expected_davies_bouldin = metrics.davies_bouldin_score(embedding, labels)
            # Print out the variables for debugging
            print("Embedding:", embedding)
            print("Labels:", labels)
            print("Expected silhouette:", expected_silhouette)
            # Check that the actual and expected scores are close
            self.assertAlmostEqual(actual_silhouette, expected_silhouette, places=2)
            self.assertAlmostEqual(
                actual_calinski_harabasz, expected_calinski_harabasz, places=2
            )
            self.assertAlmostEqual(
                actual_davies_bouldin, expected_davies_bouldin, places=2
            )


if __name__ == "__main__":
    unittest.main()
