```bash
mlflow run . -e ml_kmeans_client_value --experiment-name="Proof-of-concept KMeans" -P n_klusters=7 -P file_path=../data/20220412-171427_dataset.csv -P max_iter=500
```