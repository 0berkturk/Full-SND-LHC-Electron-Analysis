import numpy as np
import matplotlib.pyplot as plt
import os

def plot_loss(loss,title1,outname):
    min_value = np.min(loss)
    min_index = np.argmin(loss)+1
    plt.plot(np.arange(1, len(loss) + 1),loss)
    plt.axvline(x=min_index, color='red', linestyle='--', label='Best Epoch')
    #plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title1)
    plt.legend()
    plt.savefig(outname+"png")
    print(title1,min_value)
    plt.clf()

def plot_losses(train, val):
    min_index = np.argmin(val) + 1  # +1 because epochs start from 1

    # Create figure and primary y-axis
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Create secondary y-axis
    ax2 = ax1.twinx()

    # Plot training and validation loss
    epochs = np.arange(1, len(train) + 1)
    train_line, = ax1.plot(epochs, train, color="blue", label="Training Loss")

    if (os.path.exists("results/v_n_miscl_pr.npy")):
        label_v="Val. Loss(-Proton Rej.)"
    else:
        label_v = "Validation Loss"

    val_line, = ax2.plot(epochs, val, color="orange", label=label_v)

    # Best epoch marker
    best_epoch_line = ax1.axvline(x=min_index, color='red', linestyle='--', label="Best Epoch")

    # Labels
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color="blue")
    ax2.set_ylabel("Validation Loss", color="orange")

    # Title
    plt.title("Loss vs. Epoch")

    # Combine legends
    lines = [train_line, val_line, best_epoch_line]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    # Save and clear
    plt.savefig("results/all_losses.png")
    plt.clf()


def plot_prej_losses():
    p_rej = np.load("results/v_mean_loss_val.npy", allow_pickle=True)

    n_misc_pr = np.load("results/v_n_miscl_pr.npy", allow_pickle=True)
    print(len(p_rej), len(n_misc_pr))
    print(p_rej)
    print(n_misc_pr)
    if os.path.exists("results/v_bceloss.npy"):
        v_loss = np.load("results/v_bceloss.npy", allow_pickle=True)
        label_v = "BCE Loss"
    else:
        v_loss = np.load("results/v_fcloss.npy", allow_pickle=True)
        label_v = "FC Loss"

    x = np.arange(1, len(p_rej) + 1)

    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)
    # Create secondary y-axis (right)
    twin1 = ax.twinx()
    twin2 = ax.twinx()

    twin2.spines.right.set_position(("axes", 1.2))

    # Plot data
    p1, = ax.plot(x, p_rej, "g-", linestyle="-", label="Proton Rejection at 90% e.eff.")
    p2, = twin1.plot(x, n_misc_pr, "b--", linestyle="--", label="Number of Misclass. Protons wrt. electrons")
    p3, = twin2.plot(x, v_loss, "r:", linestyle=":", label=label_v)

    # Set labels
    ax.set_ylabel("Proton Rejection at 90% e.eff.", color="g")
    twin1.set_ylabel("Number of Misclass. Protons wrt. electrons", color="b")
    twin2.set_ylabel(label_v, color="r")
    ax.set_xlabel("Epoch")

    plt.title("Loss vs. Epoch")

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    ax.legend(handles=[p1, p2, p3], loc="upper center", fontsize=7,framealpha=0.6)
    for label in ax.get_legend().get_texts():
        label.set_alpha(0.5)
    # Save and clear
    plt.savefig("results/proton_Rej_val_losses.png", dpi=300)
    plt.clf()


training_loss = np.load("results/training_loss.npy",allow_pickle=True)
validation_loss = np.load("results/v_mean_loss_val.npy",allow_pickle=True)

plot_loss(training_loss,"Training Loss","results/training")
plot_loss(validation_loss,"Validation Loss","results/validation")
plot_losses(training_loss,validation_loss)

if (os.path.exists("results/v_n_miscl_pr.npy")):
    plot_prej_losses()