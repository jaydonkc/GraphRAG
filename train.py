import argparse
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx


def build_toy_graph():
    G = nx.karate_club_graph()
    x = torch.eye(G.number_of_nodes(), dtype=torch.float)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    y = torch.zeros(G.number_of_nodes(), dtype=torch.long)
    y[G.nodes[0]['club'] == 'Mr. Hi'] = 0
    y[G.nodes[0]['club'] != 'Mr. Hi'] = 1
    return Data(x=x, edge_index=edge_index, y=y)


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x


def train(data, epochs, device):
    model = GCN(data.num_node_features, 16, 2).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph = build_toy_graph()
    train(graph, args.epochs, device)
    print("Training complete on", device)

