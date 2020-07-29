# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd

from fclib.feature_engineering.dummyvar import (
    get_day_dummies,
    get_dayofyear_dummies,
    get_hour_dummies,
    get_minute_dummies,
    get_month_dummies,
    get_quarter_dummies,
    get_weekday_dummies,
    get_weekofyear_dummies,
)

date = pd.to_datetime(pd.Series(["2000-01-01", "2000-01-02", "2000-01-03"]))

time = pd.to_datetime(pd.Series(["13:00:00", "13:01:00", "13:02:00"]))


def test_get_day_dummies():
    mat = get_day_dummies(date)
    col = date.dt.day
    assert len(mat.columns) == 31
    for i in range(len(date)):
        assert mat.iloc[i, col.iloc[i] - 1] == 1
    assert mat.loc[0, 1] == 1


def test_get_dayofyear_dummies():
    mat = get_dayofyear_dummies(date)
    col = date.dt.dayofyear
    assert len(mat.columns) == 366
    for i in range(len(date)):
        assert mat.iloc[i, col.iloc[i] - 1] == 1
    assert mat.loc[0, 1] == 1


def test_get_hour_dummies():
    mat = get_hour_dummies(time)
    col = time.dt.hour
    assert len(mat.columns) == 24
    for i in range(len(time)):
        assert mat.iloc[i, col.iloc[i]] == 1
    assert mat.loc[0, 13] == 1


def test_get_minute_dummies():
    mat = get_minute_dummies(time)
    col = time.dt.minute
    assert len(mat.columns) == 60
    for i in range(len(time)):
        assert mat.iloc[i, col.iloc[i]] == 1
    assert mat.loc[0, 0] == 1


def test_get_month_dummies():
    mat = get_month_dummies(date)
    col = date.dt.month
    assert len(mat.columns) == 12
    for i in range(len(date)):
        assert mat.iloc[i, col.iloc[i] - 1] == 1
    assert mat.loc[0, "Jan"] == 1


def test_get_quarter_dummies():
    mat = get_quarter_dummies(date)
    col = date.dt.quarter
    assert len(mat.columns) == 4
    for i in range(len(date)):
        assert mat.iloc[i, col.iloc[i] - 1] == 1
    assert mat.loc[0, 1] == 1


def test_get_weekday_dummies():
    mat = get_weekday_dummies(date)
    col = date.dt.weekday
    assert len(mat.columns) == 7
    for i in range(len(date)):
        assert mat.iloc[i, col.iloc[i]] == 1
    assert mat.loc[0, "Sat"] == 1


def test_get_weekofyear_dummies():
    mat = get_weekofyear_dummies(date)
    col = date.dt.weekofyear
    assert len(mat.columns) == 53
    for i in range(len(date)):
        assert mat.iloc[i, col.iloc[i] - 1] == 1
    assert mat.loc[0, 52] == 1
