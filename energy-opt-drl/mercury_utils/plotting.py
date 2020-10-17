import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams["font.family"] = "Times New Roman"


def plot_edc_utilization(edc_file_path):
    data = pd.read_csv(edc_file_path, index_col=False)
    edc_ids = data['dc_id'].unique().tolist()
    for edc_id in edc_ids:
        x = data[data['dc_id'] == edc_id]
        plt.step(x['time'], x['std_u'], where='post')

    plt.xlabel('time [s]', fontsize=12)
    plt.ylabel('std_u factor [%]', fontsize=12)
    plt.title('Edge Data Centers Utilization Factor')
    plt.legend(edc_ids, prop={'size': 12})
    plt.show()

    graph_data = {
        'xlabel': 'time [s]',
        'ylabel': 'std_u factor [%]',
        'title': 'Total Edge Federation Utilization Factor'
    }

    sum_graph(data, 'time', 'std_u', 'dc_id', graph_data)


def plot_edc_power(edc_file_path):
    data = pd.read_csv(edc_file_path, index_col=False)
    edc_ids = data['dc_id'].unique().tolist()
    for edc_id in edc_ids:
        x = data[data['dc_id'] == edc_id]
        plt.step(x['time'], x['power'], where='post')

    plt.xlabel('time [s]', fontsize=12)
    plt.ylabel('power [W]', fontsize=12)
    plt.title('Edge Data Centers Power Consumption')
    plt.legend(edc_ids, prop={'size': 12})
    plt.show()

    graph_data = {
        'xlabel': 'time [s]',
        'ylabel': 'power [W]',
        'title': 'Total Edge Federation Power Consumption'
    }

    sum_graph(data, 'time', 'power', 'dc_id', graph_data)


def sum_graph(data, x_column, y_column, class_column, graph_data):
    class_labels = data[class_column].unique().tolist()
    n_labels = len(class_labels)
    x = data[x_column].values.tolist()

    data_array = np.zeros((len(x), len(class_labels)))
    for i in range(n_labels):
        last_value = 0
        for index, row in data.iterrows():
            if row[class_column] == class_labels[i]:
                increment = row[y_column] - last_value
                data_array[index:, i:n_labels] += increment
                last_value = row[y_column]

    plt.step(x, data_array[:, n_labels - 1], where='post')
    plt.xlabel(graph_data['xlabel'], fontsize=12)
    plt.ylabel(graph_data['ylabel'], fontsize=12)
    plt.title(graph_data['title'])
    plt.show()


def multiple_graph(dataframes, x_column, y_column, class_column, graph_data):
    color = ['r', 'limegreen']
    j = 0
    for data in dataframes:
        class_labels = data[class_column].unique().tolist()
        n_labels = len(class_labels)
        x = data[x_column].values.tolist()

        data_array = np.zeros((len(x), len(class_labels)))
        for i in range(n_labels):
            last_value = 0
            for index, row in data.iterrows():
                if row[class_column] == class_labels[i]:
                    increment = row[y_column] - last_value
                    data_array[index:, i:n_labels] += increment
                    last_value = row[y_column]

        plt.step(x, data_array[:, n_labels - 1], where='post', color=color[j])
        j += 1
    plt.xlabel(graph_data['xlabel'], fontsize=12)
    plt.ylabel(graph_data['ylabel'], fontsize=12)
    plt.legend(graph_data['legend'], prop={'size': 12})
    plt.title(graph_data['title'])
    plt.show()


def delay_graph(dataframes, x_column, y_column, graph_data):
    color = ['limegreen', 'r']
    j = 0
    for data in dataframes:
        plt.plot(data[x_column], data[y_column], color=color[j])
        j += 1
    plt.xlabel(graph_data['xlabel'], fontsize=12)
    plt.ylabel(graph_data['ylabel'], fontsize=12)
    plt.legend(graph_data['legend'], prop={'size': 12})
    plt.title(graph_data['title'])
    plt.show()


def delay_ema_graph(dataframes, x_column, y_column, graph_data, alpha=0.05):
    color = ['limegreen', 'r']
    j = 0
    for data in dataframes:
        ema = np.copy(data[y_column].values)
        for i in range(1, ema.shape[0]):
            ema[i] = (1 - alpha) * ema[i - 1] + alpha * ema[i]
        plt.plot(data[x_column], ema, color=color[j])
        j += 1
    plt.xlabel(graph_data['xlabel'], fontsize=12)
    plt.ylabel(graph_data['ylabel'], fontsize=12)
    plt.legend(graph_data['legend'], prop={'size': 12})
    plt.title(graph_data['title'])
    plt.show()


def delay_summary(ue_file_path, t_threshold=None, max_d=None, min_d=None):
    data = pd.read_csv(ue_file_path, sep=';')
    if t_threshold is not None:
        data = data[data['time'] >= t_threshold]
    if max_d is not None:
        data = data[data['delay'] <= max_d]
    if min_d is not None:
        data = data[data['delay'] >= min_d]
    delay = data['delay'].values
    mean_delay = np.mean(delay)
    peak_delay = np.max(delay)
    print("Mean delay: {} seconds".format(mean_delay))
    print("Peak delay: {} seconds".format(peak_delay))
    return mean_delay, peak_delay


def power_summary(edc_file_path, t_threshold=None):
    data = pd.read_csv(edc_file_path, index_col=False, sep=';')
    if t_threshold is not None:
        data = data[data['time'] >= t_threshold]
    else:
        t_threshold = 0
    edc_ids = data['edc_id'].unique().tolist()
    n_edcs = len(edc_ids)
    max_power = 0
    t_prev = 0
    power = np.zeros(n_edcs)
    mean_power = 0
    for index, row in data.iterrows():
        mean_power += np.sum(power) * (row['time'] - t_prev)
        t_prev = row['time']
        for i in range(n_edcs):
            if row['edc_id'] == edc_ids[i]:
                power[i] = row['overall_power']
        max_power = max(max_power, np.sum(power))
    mean_power /= (t_prev - t_threshold)
    print("Mean power: {} Watts".format(mean_power))
    print("Peak power: {} Watts".format(max_power))
    return mean_power, max_power


if __name__ == '__main__':
    standby = list(range(11))
    strategies = {'emptiest': 'min', 'fullest': 'max'}
    try:
        ues = [i * 10 for i in range(1, 11)]
        df = pd.read_csv('./res/resume.csv', sep=';')
    except:
        filepath_root = './res'
        df = pd.DataFrame(columns=['strategy', 'standby', 'n_ues', 'srv_mean', 'srv_peak',
                                   'start_mean', 'd_mean', 'd_peak', 'p_mean', 'p_peak'], index=None)
        for strategy, f_1 in strategies.items():
            for hot in standby:
                for n_ues in ues:
                    filepath = '/'.join([filepath_root, f_1, str(hot), str(n_ues)])
                    print("****************")
                    print(filepath)
                    print("****************")
                    delay_filepath = '/'.join([filepath, 'ue_report.csv'])
                    power_filepath = '/'.join([filepath, 'edc_report.csv'])
                    d_mean, d_peak = delay_summary(delay_filepath)
                    srv_mean, srv_peak = delay_summary(delay_filepath, max_d=0.2)
                    start_mean, _ = delay_summary(delay_filepath, min_d=0.2)
                    p_mean, p_peak = power_summary(power_filepath)

                    df = df.append({'strategy': strategy,
                                    'standby': hot,
                                    'n_ues': n_ues,
                                    'srv_mean': srv_mean,
                                    'srv_peak': srv_peak,
                                    'start_mean': start_mean,
                                    'd_mean': d_mean,
                                    'd_peak': d_peak,
                                    'p_mean': p_mean,
                                    'p_peak': p_peak,
                                    }, ignore_index=True)

        df.to_csv('./res/resume.csv', sep=';')


    fig, axs = plt.subplots(4, 2, sharey='row', figsize=(9, 10), gridspec_kw={'height_ratios': [17, 17, 17, 1]})
    axs[3, 0].axis('off')
    axs[3, 1].axis('off')

    d1 = axs[0, 0]
    d2 = axs[0, 1]
    m1 = axs[1, 0]
    m2 = axs[1, 1]
    p1 = axs[2, 0]
    p2 = axs[2, 1]

    d1.get_shared_x_axes().join(d1, m1, p1)
    d2.get_shared_x_axes().join(d2, m2, p2)
    m1.get_shared_x_axes().join(d1, m1, p1)
    m2.get_shared_x_axes().join(d2, m2, p2)
    p1.get_shared_x_axes().join(d1, m1, p1)
    p2.get_shared_x_axes().join(d2, m2, p2)
    #d1.set_xticklabels([])
    #d2.set_xticklabels([])
    #m1.set_xticklabels([])
    #m2.set_xticklabels([])

    d1.set_title('a) Mean Perceived Delay (emptiest)', size=14)
    d2.set_title('b) Mean Perceived Delay (fullest)', size=14)
    m1.set_title('c) Mean Power Consumption (emptiest)', size=14)
    m2.set_title('d) Mean Power Consumption (fullest)', size=14)
    p1.set_title('e) Peak Power Consumption (emptiest)', size=14)
    p2.set_title('f) Peak Power Consumption (fullest)', size=14)

    markers = {
        0: '.',
        1: '2',
        2: 'x',
        3: 'o',
        4: '^',
        5: 's',
        6: 'p',
        7: '*',
        8: 'd',
        9: 'P',
        10: 'X'
    }
    # first column
    data = df[df['strategy'] == 'emptiest']
    for hot in standby:
        that = data[data['standby'] == hot]
        markersize = 9 if hot in [0, 1, 6, 7] else 6
        d1.plot(that['n_ues'], that['d_mean'], marker=markers[hot], linestyle='--', label=hot, markersize=markersize)
        m1.plot(that['n_ues'], that['p_mean'], marker=markers[hot], linestyle='--', label=hot, zorder=(10 - hot) * 5, markersize=markersize)
        p1.plot(that['n_ues'], that['p_peak'], marker=markers[hot], linestyle='--', label=hot, zorder=(10 - hot) * 5, markersize=markersize)

    # second column
    data = df[df['strategy'] == 'fullest']
    for hot in standby:
        that = data[data['standby'] == hot]
        markersize = 9 if hot in [0, 1, 6, 7] else 6
        d2.plot(that['n_ues'], that['d_mean'], marker=markers[hot], linestyle='--', label=hot, markersize=markersize)
        m2.plot(that['n_ues'], that['p_mean'], marker=markers[hot], linestyle='--', label=hot, zorder=(10 - hot) * 5, markersize=markersize)
        p2.plot(that['n_ues'], that['p_peak'], marker=markers[hot], linestyle='--', label=hot, zorder=(10 - hot) * 5, markersize=markersize)

    handles, labels = p1.get_legend_handles_labels()
    l = fig.legend(handles, labels, loc='lower center', prop={'size': 11}, ncol=11, mode='expand',
               bbox_to_anchor=(0.07, 0.01, 0.91, 0.2), title="Number of PUs in Hot Standby")
    l.get_title().set_fontsize(12)

    # fig.text(0.55, 0.01, 'UEs', ha='center')
    d1.set_ylabel('t (s)', size=12)
    m1.set_ylabel('P (W)', size=12)
    p1.set_ylabel('P (W)', size=12)

    p1.set_xlabel('UEs', size=12)
    p2.set_xlabel('UEs', size=12)

    fig.show()
