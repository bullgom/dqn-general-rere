from network import Network
import os
import json
import shutil
import torch


class Saver:
    """Handles saving of different files"""
    
    def __init__(
        self, 
        base_folder: str,
        run_name: str,
        files: list[str],
        networks: dict[str, Network]
    ):
        self.base_path = base_folder
        self.run_name = run_name
        self.files = files
        self.networks = networks
        
        self.run_folder = os.path.join(base_folder, run_name)
        
        self.best_model_path = os.path.join(self.run_folder, "best.pt")
        self.last_model_path = os.path.join(self.run_folder, "last.pt")
        self.results_path = os.path.join(self.run_folder, "results.json")
        self.state_path = os.path.join(self.run_folder, "state.json")
        self.image_path = os.path.join(self.run_folder, "learning_graph.png")
        
        self.state = {}
        
        self.results = {
            "best": -1,
            "last": -1,
        }
        
    def save_experiment(self) -> None:
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        shutil.copytree("./", self.run_folder, ignore=lambda dir, content: [self.base_path])
        
    
    def save_state(self):
        
        with open(self.results_path, "w") as fp:
            json.dump(self.results, fp, indent=4, sort_keys=True)
        
        with open(self.state_path, "w") as fp:
            json.dump(self.state, fp, indent=4, sort_keys=True)
        
        for name, network in self.networks.items():
            assert ".pt" in name
            path = os.path.join(self.run_folder, name)
            torch.save(network, path)
    