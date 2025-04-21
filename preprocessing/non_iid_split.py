import pandas as pd
import numpy as np

def non_iid_split(df, n_clients, alpha=0.5, min_samples_per_client=1):
    classes = np.unique(df['Class'])
    proportions = np.random.dirichlet([alpha] * n_clients, len(classes))
    client_indices = [[] for _ in range(n_clients)]

    for class_id in classes:
        class_indices = np.unique(df[df['Class'] == class_id]['SubjectID'])
        n_class_samples = len(class_indices)

        # determine number of samples per client for given class
        class_proportions = proportions[class_id]
        samples_per_client = (class_proportions * n_class_samples).astype(int)

        # ensures no client node is given 0 samples
        for i in range(n_clients):
            if samples_per_client[i] < min_samples_per_client and n_class_samples > 0:
                samples_per_client[i] = min(min_samples_per_client, n_class_samples)

        # ensures # of samples per client match total samples
        total_assigned = np.sum(samples_per_client)
        if total_assigned < n_class_samples:
            samples_per_client[np.argmax(class_proportions)] += n_class_samples - total_assigned
        elif total_assigned > n_class_samples:
            samples_per_client[np.argmax(class_proportions)] -= total_assigned - n_class_samples

        # ensuring sample distribution is random and non-uniform
        np.random.shuffle(class_indices)
        split_points = np.cumsum(samples_per_client)[:-1] # omit last value, creates a nth client with 0 samples
        class_splits = np.split(class_indices, split_points)

        # assign to clients - do for both non-pd and pd patients
        for client_id, indices in enumerate(class_splits):
            client_indices[client_id].extend(indices)

    # shuffle with clients grouped
    for client_id in range(n_clients):
        np.random.shuffle(client_indices[client_id])

    return client_indices

df = pd.read_csv('../labelled_patients.csv')
client_indices = non_iid_split(df, 5)

for i, indices in enumerate(client_indices):
    client_labels = df['Class'][df['SubjectID'].isin(indices)]
    n_parkinsons = np.sum(client_labels)
    n_healthy = len(client_labels) - n_parkinsons

    print(f"Client: {i+1}: {len(client_labels)} individuals, "
          f"{n_parkinsons} Parkinson's, {n_healthy} healthy, "
          f"Parkinson's ratio: {n_parkinsons / len(client_labels):.2f}")

