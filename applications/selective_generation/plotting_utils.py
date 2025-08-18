import matplotlib.pyplot as plt

def plot_risk_coverage(rc, output_file):
    """Plot Risk–Coverage curve from compute_risk_coverage result."""
    plt.figure()
    plt.plot(rc.coverage, rc.risk, label=f"AURC={rc.aurc:.3f}")
    plt.xlabel("Coverage")
    plt.ylabel("Risk (Error Rate)")
    plt.title("Risk–Coverage Curve")
    plt.legend()
    plt.show()
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

def plot_accuracy_coverage(rc, output_file):
    """Plot Accuracy–Coverage curve from compute_risk_coverage result."""
    plt.figure()
    plt.plot(rc.coverage, rc.accuracy)
    plt.xlabel("Coverage")
    plt.ylabel("Accuracy")
    plt.title("Accuracy–Coverage Curve")
    plt.show()
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

def plot_reliability(rel, output_file):
    """Plot reliability diagram from reliability_diagram result."""
    import matplotlib.ticker as mticker
    plt.figure()
    width = 1.0 / len(rel.bin_centers)
    # observed accuracy bars
    plt.bar(rel.bin_centers, rel.bin_accuracy, width=0.9*width, alpha=0.6, label="Observed Accuracy")
    # predicted confidence line
    plt.plot(rel.bin_centers, rel.bin_confidence, marker="o", label="Predicted Confidence")
    # diagonal
    plt.plot([0,1],[0,1],"--", color="black")
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(11))
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Observed Accuracy")
    plt.title(f"Reliability Diagram (ECE={rel.ece:.3f})")
    plt.legend()
    plt.show()
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
