import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


def collater(data_list):
    batch = {}
    name = [x["name"] for x in data_list]
    protac_graphs = [x["protac_graphs"] for x in data_list]
    ligase_pocket = [x["ligase_pocket"] for x in data_list]
    target_pocket = [x["target_pocket"] for x in data_list]
    feature = [x['feature'] for x in data_list]
    label = [x["label"] for x in data_list]

    batch["name"] = name
    batch["protac_graphs"] = Batch.from_data_list(protac_graphs)
    batch["ligase_pocket"] = Batch.from_data_list(ligase_pocket)
    batch["target_pocket"] = Batch.from_data_list(target_pocket)
    batch["feature"] = torch.tensor(feature, dtype=torch.float)
    batch["label"]=torch.tensor(label)
    return batch


class PROTACSet(Dataset):
    def __init__(self, name, protac_graphs, ligase_pocket, target_pocket, feature, label):
        super().__init__()
        self.name = name
        self.protac_graphs = protac_graphs
        self.ligase_pocket = ligase_pocket
        self.target_pocket = target_pocket
        self.feature = feature
        self.label = label


    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        sample = {
            "name": self.name[idx],
            'protac_graphs': self.protac_graphs[idx],
            "ligase_pocket": self.ligase_pocket[idx],
            "target_pocket": self.target_pocket[idx],
            'feature': self.feature[idx],
            "label": self.label[idx],
        }
        return sample


