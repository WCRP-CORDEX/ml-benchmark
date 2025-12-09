import matplotlib.pyplot as plt

def plot_losses(loss_train, loss_test, cf):
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.plot(loss_train, label='Train Loss')
    axs.plot(loss_test, label='Test Loss')
    axs.legend()
    axs.set_yscale('log')
    axs.grid(True)
    plt.savefig(cf.logging.run_dir / "plots/losses.png", dpi=150)
    plt.close()