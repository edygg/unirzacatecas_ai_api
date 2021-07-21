import pdfkit
from datetime import datetime
from django.template import Template, Context
from django.conf import settings


TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8" />
    <title>BAIP - UAZ</title>
    <style>
        table {
          width: 80%;
          border: 1px solid #616b94;
          border-spacing: 0;
        }

        th,
        td {
          width: 25%;
          text-align: left;
          vertical-align: top;
          border-collapse: collapse;
          border: 1px solid #616b94;
          padding: 0.3em;
        }

        caption {
          padding: 0.2em;
        }

        th {
          background: #e3e7f7;
        }

        img {
          border: 0px solid #ddd;
          border-radius: 4px;
          padding: 5px;
          width: 90%;
        }
    </style>
  </head>
  <body>
    <section>
      <p>
        Estimado usuario, a continuación, encontrará los resultados para el
        entrenamiento de su conjunto de datos.
      </p>
    </section>
    <section>
      <p>
        <br />
        <b>Fecha: </b>{{ date }}<br />
        <b>ID: </b>{{ execution_id }}<br />
        <b>e-mail: </b>{{ execution_email }}<br />
        <b>Conjunto de datos: </b>{{ execution_dataset_filename }}<br />
      </p>
    </section>
    <section>
      <br />
      <h2>
        Informe de resultados
      </h2>
      <p>
        <b>Tipo de algoritmo: </b>{{ algorithm_type }}<br />
        <b>Algoritmos ejecutados: </b>{{ algorithms_list }}<br />
        <b>Variable objetivo: </b>{{ target_colname }}<br />
        <b>Variables dependientes: </b>{{ dependent_colname_list }}<br />
        <b>Hold-out: </b>{{ hold_out }}%<br />
        <b>Cross-validation: </b>{{ cross_validation }}<br />
      </p>
    </section>
    <section>
      <br />
      <h3>
        Resultados por algoritmo
      </h3>
    </section>
    {{ partial_html_results_by_algorithm|safe}}
'''



def generate_pdf_report(algorithm_results, training_run):
    dateTimeObj = datetime.now()
    date = dateTimeObj.strftime('%d-%b-%Y - %H:%M:%S.%f')

    execution_id = 787600123231
    execution_email = 'xxx@ddd.com'
    execution_dataset_filename = 'file123.csv'
    algorithm_type = 'Regresión'
    algorithms_list = ['Árboles de decisión', 'Redes neuronales']
    target_colname = 'col_objetivo'
    dependent_colname_list = ['columna2', 'columna3', 'columna4', 'columna6', 'columna7', 'columna8', 'columna9',
                              'columna10', 'columna12']
    hold_out = 0.3
    cross_validation = 4

    t = Template(TEMPLATE)
    c = Context({'date': date,
                 'execution_id': execution_id,
                 'execution_email': execution_email,
                 'execution_dataset_filename': execution_dataset_filename,
                 'algorithm_type': algorithm_type,
                 'algorithms_list': algorithms_list,
                 'target_colname': target_colname,
                 'dependent_colname_list': dependent_colname_list,
                 'cross_validation': cross_validation,
                 'partial_html_results_by_algorithm': str(algorithm_results), })
    final_html = t.render(c)

    options = {
        'dpi': '300',
        'page-size': 'A4',
        'margin-top': '45mm',
        'margin-right': '25mm',
        'margin-bottom': '25mm',
        'margin-left': '25mm',
        'encoding': "UTF-8",
        'no-outline': None,
        'header-html': 'https://marian-testing2.glitch.me/header.html',
        'footer-right': 'Página [page] de [topage]',
        'header-spacing': '6',
        'footer-spacing': '4'
    }

    pdfkit.from_string(str(final_html), f'{settings.MEDIA_ROOT}/execution-{training_run.id}.pdf', options=options)
