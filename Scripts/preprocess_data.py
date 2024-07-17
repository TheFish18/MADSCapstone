import pandas as pd

def parse_header(row):
    """ parse historical calendar data """
    row = row.split(' ')
    n = len(row)
    if n == 5 or n == 8:
        month, day, year = row[0], row[1], row[-1]
        return month.split('/')[0], day.split('-')[0], year
    elif n == 7:
        month, day, year = row[0], row[1].split('-')[0], row[-1]
        return month.split('/')[0], day.split('-')[0], year
    else:
        return None, None, None


def combine_date():
    p_funds = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/FEDFUNDS.csv"
    p_fomc = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/fomc_calendar.csv"
    p_sp500 = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/sp500.csv"

    df_fomc = pd.read_csv(p_fomc)
    df_fomc['date'] = pd.to_datetime(df_fomc['date'])

    df_funds = pd.read_csv(
        p_funds
    ).rename(
        columns={"DATE": "date", "FEDFUNDS": "fedfunds"}
    )
    df_funds['date'] = pd.to_datetime(df_funds['date'])

    df_sp500 = pd.read_csv(
        p_sp500
    ).rename(
        columns={'Date': "date", "Close": "close"}
    )
    df_sp500['date'] = pd.to_datetime(df_sp500['date'])

    def get_closest_date(row):
        idxmin = (row['date'] - df_funds['date']).abs().idxmin()
        fedfund = df_funds['fedfunds'].iloc[idxmin]
        return fedfund

    df = df_fomc[['date']].merge(df_sp500[['date', 'close']], on='date', how='outer')
    df['fedfunds'] = df.apply(get_closest_date, axis=1)

    p = "/Users/joshfisher/PycharmProjects/MADSCapstone/Data/fomc_fed_sp500.csv"
    df.to_csv(p, index=False)


def fomc_impact(df_sp500, df_fomc, save_p):
    df_sp500.date = pd.to_datetime(df_sp500.date)
    close = df_sp500.close
    df_fomc.date = pd.to_datetime(df_fomc.date)

    def tmp(row):
        og_idx = row['index']
        idx = (row[0] - df_sp500.date).abs().idxmin()
        val = (close.iloc[idx+1: idx+6].mean() - close.iloc[idx-5: idx].mean())
        val_norm = val / close.iloc[idx-5: idx].mean()
        return og_idx, val, val_norm

    dates = pd.Series(
        df_fomc.date.unique()
    ).sort_values(
        ascending=False
    ).reset_index()

    diffs = dates.apply(
        tmp, axis=1, result_type="expand"
    ).rename(
        columns={0: 'index', 1: "diff", 2: "diff_norm"}
    )

    df = dates.merge(
        diffs, on='index', how='outer'
    ).drop(
        columns=['index']
    ).rename(
        columns={0: 'date'}
    )

    df = df.sort_values(by='date', ascending=False)
    df.to_csv(save_p, index=False)


def parse_beige_book_dates(beige_book_path: str):
    df = pd.read_csv(beige_book_path)

    def tmp(row):
        row = row.split('\\n')
        date1 = row[0].strip("\"\'").replace(' \t', '')
        date2 = row[1].strip("\"\'").replace(' \t', '')
        dates = [date1, date2]
        for date in dates:
            try:
                return pd.to_datetime(date, format='%B %d, %Y')
            except ValueError:
                continue
        return None

    df.insert(2, 'date2', df.text.apply(tmp))
    df.to_csv(beige_book_path, index=False)


def merge_beige_books_impact(df_beige_book: pd.DataFrame, df_fomc_impact: pd.DataFrame):
    """
    The beige book comes out about two weeks before the FOMC.
    Grabs the beige book and fomc impact pair
    Args:
        df_bb:
        df_impact:

    Returns:
    """
    rows = []
    df_beige_book = df_beige_book.reset_index()
    df_fomc_impact = df_fomc_impact.reset_index()

    time_delta = pd.to_timedelta(14., unit="D")
    for bb_date in df_beige_book.date.unique():
        est_im_date = bb_date + time_delta
        date_diffs = (est_im_date - df_fomc_impact.date).abs()
        idxmin = date_diffs.idxmin()
        impact = df_fomc_impact.iloc[idxmin]
        impact['bb_date'] = bb_date
        rows.append(impact)

    tmp = pd.DataFrame.from_records(
        rows
    ).rename(
        columns={'date': 'impact_date'}
    )

    df = df_beige_book.merge(
        tmp, left_on='date', right_on='bb_date'
    ).drop(
        columns=['date', 'index_x', 'index_y']
    )

    return df


if __name__ == "__main__":
    p_sp500 = "../../Data/sp500.csv"
    p_fomc = "../../Data/fomc_dates.csv"
    p_impact = "../../Data/fomc_impact.csv"

    df_sp500 = pd.read_csv(p_sp500)
    df_fomc = pd.read_csv(p_fomc)

    fomc_impact(df_sp500, df_fomc, p_impact)
