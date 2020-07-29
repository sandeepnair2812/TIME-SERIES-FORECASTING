# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import calendar
import pandas as pd
import numpy as np


def get_month_dummies(date):
    """
    Generate matrix of dummy variables for the month of the year.

    Args:
        date: Datetime Pandas series.

    Returns:
        Dataframe with 12 columns containing dummy variables. The column names are the month abbreviations.
    """
    full = pd.DataFrame(np.zeros((len(date), 12)), dtype="uint8", columns=pd.RangeIndex(1, 13))
    mon = pd.get_dummies(date.dt.month)
    full[mon.columns] = mon
    full.columns = calendar.month_abbr[slice(1, 13)]
    return full


def get_weekofyear_dummies(date):
    """
    Generate matrix of dummy variables for the week of the year.

    Args:
        date: Datetime Pandas series.

    Returns:
        Dataframe with 53 columns containing dummy variables. The columns are labelled from 1 to 53.
    """
    full = pd.DataFrame(np.zeros((len(date), 53)), dtype="uint8", columns=pd.RangeIndex(1, 54))
    wk = pd.get_dummies(date.dt.weekofyear)
    full[wk.columns] = wk
    return full


def get_dayofyear_dummies(date):
    """
    Generate matrix of dummy variables for the day of the year.

    Args:
        date: Datetime Pandas series.

    Returns:
        Dataframe with 366 columns containing dummy variables. The columns are labelled from 1 to 366.
    """
    full = pd.DataFrame(np.zeros((len(date), 366)), dtype="uint8", columns=pd.RangeIndex(1, 367))
    dyr = pd.get_dummies(date.dt.dayofyear)
    full[dyr.columns] = dyr
    return full


def get_day_dummies(date):
    """
    Generate matrix of dummy variables for the day of the month.

    Args:
        date: Datetime Pandas series.

    Returns:
        Dataframe with 31 columns containing dummy variables. The columns are labelled from 1 to 31.
    """
    full = pd.DataFrame(np.zeros((len(date), 31)), dtype="uint8", columns=pd.RangeIndex(1, 32))
    day = pd.get_dummies(date.dt.day)
    full[day.columns] = day
    return full


def get_weekday_dummies(date):
    """
    Generate matrix of dummy variables for the day of the week.

    Args:
        date: Datetime Pandas series.

    Returns:
        Dataframe with 7 columns containing dummy variables. The column names are the weekday abbreviations.
    """
    full = pd.DataFrame(np.zeros((len(date), 7)), dtype="uint8", columns=pd.RangeIndex(0, 7))
    wkd = pd.get_dummies(date.dt.weekday)
    full[wkd.columns] = wkd
    full.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return full


def get_quarter_dummies(date):
    """
    Generate matrix of dummy variables for the quarter of the year.

    Args:
        date: Datetime Pandas series.

    Returns:
        Dataframe with 4 columns containing dummy variables. The columns are labelled from 1 to 4.
    """
    full = pd.DataFrame(np.zeros((len(date), 4)), dtype="uint8", columns=pd.RangeIndex(1, 5))
    qt = pd.get_dummies(date.dt.quarter)
    full[qt.columns] = qt
    return full


def get_hour_dummies(date):
    """
    Generate matrix of dummy variables for the hour of the day.

    Args:
        date: Datetime Pandas series.

    Returns:
        Dataframe with 24 columns containing dummy variables. The columns are labelled from 0 to 23.
    """
    full = pd.DataFrame(np.zeros((len(date), 24)), dtype="uint8", columns=pd.RangeIndex(0, 24))
    hr = pd.get_dummies(date.dt.hour)
    full[hr.columns] = hr
    return full


def get_minute_dummies(date):
    """
    Generate matrix of dummy variables for the minute of the hour.

    Args:
        date: Datetime Pandas series.

    Returns:
        Dataframe with 60 columns containing dummy variables. The columns are labelled from 0 to 59.
    """
    full = pd.DataFrame(np.zeros((len(date), 60)), dtype="uint8", columns=pd.RangeIndex(0, 60))
    min = pd.get_dummies(date.dt.minute)
    full[min.columns] = min
    return full
