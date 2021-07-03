import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
import dominate
from dominate.tags import *
from dominate.util import raw
from sklearn import metrics


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          execution_id=''):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title, loc='left', size=18, pad=22)
    plt.savefig(execution_id + '_matriz_conf.png')


def classification_metrics(algorithm_type, categories, y_test, predictions, grid_search_models):
    try:
        str_categories = [str(category) for category in categories]
    except:
        str_categories = categories

    dict_best_params = eval(str(grid_search_models.best_params_))
    df_best_params = pd.Series(dict_best_params)
    df_best_params = pd.DataFrame(df_best_params)
    df_best_params.reset_index(inplace=True)
    df_best_params.rename({'index': 'Parámetro', 0: 'Valor', }, axis='columns', inplace=True)
    html_best_params = df_best_params.to_html(index=False)

    report = classification_report(y_test, predictions, target_names=str_categories, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report.reset_index(inplace=True)
    df_classification_report.rename({'index': 'Categorias', }, axis='columns', inplace=True)
    html_txt_report = df_classification_report.to_html(index=False)

    TITLE_BEST_PARAMS = 'Mejores parametros'
    TITLE_REPORT = 'Reporte de clasificacion'

    # Recorrer para cada algoritmos
    path_matrix_img = "https://cdn.glitch.com/770c74ae-47b2-42d3-81b0-64d130c4d2f6%2F453453453_matriz_conf.png?v=1623972060069"  # Dinamico

    html_txt_title_alg = div(h3(algorithm_type))
    html_txt_best = div(p(h3(TITLE_BEST_PARAMS)), p(raw(html_best_params)))
    html_txt_matrix = div(img(src=path_matrix_img, alt="confusion_matrix"))
    html_txt_rep = div(h4(TITLE_REPORT), raw(html_txt_report))

    return html_txt_title_alg + html_txt_best + html_txt_matrix + html_txt_rep


def regression_metrics(real, predicted):
    TITLE_REPORT = 'Metricas de regresion'

    mean_squared_error = metrics.mean_squared_error(real, predicted)

    ME = "Error Maximo: {:.2f}".format(metrics.max_error(real, predicted))
    MAE = "Error Absoluto Medio (MAE): {:.2f}".format(metrics.mean_absolute_error(real, predicted))
    MSE = "Error Cuadratico Medio (MSE): {:.2f}".format(mean_squared_error)
    RMSE = "Raiz del Error Cuadratico Medio (RMSE): {:.2f}".format(mean_squared_error ** 0.5)
    RMLSE = "Logaritmo de la Raiz del Error Cuadratico Medio (RMLSE): {:.4f}".format(
        metrics.mean_squared_log_error(real, predicted))
    R2 = "Coeficiente de Determinacion R2 (r2 score): {:.4f}".format(metrics.r2_score(real, predicted))

    html_txt_title_met = section(p(h3(TITLE_REPORT)))
    html_txt_met = section(p(R2), p(MAE), p(MSE), p(RMSE), p(RMLSE), p(ME))
    html_txt = div(p(html_txt_title_met), p(html_txt_met))
    # print(html_txt)
    return html_txt

