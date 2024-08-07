import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings, os
warnings.filterwarnings("ignore")


def get_risk_score(model, predict_score, *args, **kwargs):

    if not hasattr(model, predict_score):
        raise AttributeError(f"The model does not have a method named '{predict_score}'")
    
    method = getattr(model, predict_score)
    
    if not callable(method):
        raise AttributeError(f"'{predict_score}' is not a callable method of the model")
    
    return method(*args, **kwargs)


def plot_km(dataset, model, func, plot_info, *args, **kwargs)

    dataset['risk_score'] = get_risk_score(model, func, dataset)
    median_risk_score = dataset['risk_score'].median()
    group1 = dataset[dataset['risk_score'] <= median_risk_score]
    group2 = dataset[dataset['risk_score'] > median_risk_score]

    logrank_results = logrank_test(
        durations_A=group1['survival_days'],
        durations_B=group2['survival_days'],
        event_observed_A=group1['outcome'],
        event_observed_B=group2['outcome']
    )

    kmf_train = KaplanMeierFitter()
    plt.figure(figsize=(6, 4))

    kmf_train.fit(durations=group1['survival_days'], event_observed=group1['outcome'], label='Low risk')
    kmf_train.plot_survival_function()

    kmf_train.fit(durations=group2['survival_days'], event_observed=group2['outcome'], label='High risk')
    kmf_train.plot_survival_function()

    plt.xlabel('Time (days)')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.annotate(f'p-value: {logrank_results.p_value:.3f}', xy=(0.65, 0.1), xycoords='axes fraction', fontsize=12)
    plt.savefig(f'{os.getcwd()}/km_plot_{plot_info}.png', dpi=500)