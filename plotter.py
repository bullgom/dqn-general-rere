import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Plot:
    title: str
    x_axis_title: str
    y_axis_title: str
    data: dict[str, list[float]]

class Plotter:
    
    colors = ["red", "orange", "yellow", "lime", "green", "cyan", "blue", "purple", "magenta", "grey"]
    
    def __init__(self, plots: list[Plot]) -> None:
        figure, axes = plt.subplots(len(plots), ncols=1, figsize=(10,9), constrained_layout=True)
        self.figure = figure
        self.axes = axes
        
        plt.ion()
    
    def plot(self, plots: list[Plot]) -> None:
                
        for sub_plot, axe in zip(plots, self.figure.axes):
            axe.clear()
            axe : plt.Axes
            sub_plot : Plot
            axe.set_xlabel(sub_plot.x_axis_title)
            axe.set_ylabel(sub_plot.y_axis_title)
            axe.set_title(sub_plot.title)
            
            for color, (key, data) in zip(self.colors, sub_plot.data.items()):
                axe.plot(data, label=key, color=color, linewidth = 1)
            axe.grid()                        
            axe.legend()
                
        plt.pause(0.005)
        
        