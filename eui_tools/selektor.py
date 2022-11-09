import argparse
from eui_selektor_client import EUISelektorClient, EUISelektorFormViewer
from eui_selektor_client.cli import parse_query_args
import pandas
from astropy.time import Time


def cli():
    p = argparse.ArgumentParser(
        description='Lightweight client for EUI selektor')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        '--view-form', action='store_true',
        help='retrieve and display the search form parameters')
    g.add_argument(
        '--query', metavar='param', nargs='+',
        help='run a query (parameter format: "name:value")')
    p.add_argument(
        '--output', metavar='csv_file',
        help='save the query results to a file')
    args = p.parse_args()

    if args.view_form:
        form_viewer = EUISelektorFormViewer()
        form = form_viewer.get_form()
        form_viewer.show_form(form)

    if args.query:
        client = EUISelektorClient()
        query = parse_query_args(args.query)
        final = None
        while True:
            res = client.search(query)
            if res is None:
                break
            else:
                final = pandas.concat([res, final]) if query['order[]'] == 'DESC' else pandas.concat([final, res])
                if len(res) < query['limit[]']:
                    break
            query['date_begin_start'] = res['date-beg'][0] if query['order[]'] == 'DESC' else res['date-beg'][-1]
        res_str = final.to_csv(args.output)
        if res_str is not None:
            print(res_str)
