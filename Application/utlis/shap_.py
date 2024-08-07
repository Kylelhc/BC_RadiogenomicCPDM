
import shap
import matplotlib.pyplot as plt

def plot_shap(model, X_train)
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    plt.figure()
    shap.summary_plot(shap_values.values, X_train, show=False)
    plt.show()
    plt.close()



