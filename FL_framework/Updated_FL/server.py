import fedml
from model import Deeper3DCNN
from dataset.client_dataset import load_partition_data

def load_data(args):
    return load_partition_data(args)

def create_model(args):
    return Deeper3DCNN()

if __name__ == "__main__":
    fedml.run_federated(
        args=None,
        device=None,
        dataset=load_data,
        model=create_model
    )
