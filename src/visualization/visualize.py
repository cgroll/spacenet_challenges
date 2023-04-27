import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

def show_image_and_true_false_classifications(sample, classes_masked):
    
    prg = ['#a503fc','#f50d05', '#52f705'] # purple, red, green
    my_cmap = ListedColormap(sns.color_palette(prg).as_hex())
    
    fig = plt.figure(figsize=(20,40))

    plt.subplot(1,2,1)
    plt.imshow(sample["image"].numpy().transpose(1, 2, 0))
    plt.title('Input image')

    plt.subplot(1,2,2)
    plt.imshow(sample["image"].numpy().transpose(1, 2, 0))
    plt.imshow(classes_masked, alpha=0.5, cmap=my_cmap, vmin=1, vmax=3)

    plt.title('True positives (green), false negatives (red) and positives (purple)')
    plt.show()