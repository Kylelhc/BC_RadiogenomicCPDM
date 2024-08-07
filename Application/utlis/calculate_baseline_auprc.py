

def calculate_baseline_auprc(y_true):
    P = sum(y_true)
    T = len(y_true)
    baseline_precision = P / T
    baseline_auprc = baseline_precision
    return baseline_auprc

if __name__ == "__main__":
    y_true = [0,1,0,0,1]
    baseline_auprc = calculate_baseline_auprc(y_true)
    print(f"Baseline AUPRC: {baseline_auprc:.2f}")



