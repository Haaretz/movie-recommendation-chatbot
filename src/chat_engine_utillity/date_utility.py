import datetime

import gradio as gr


def toggle_custom_date_visibility(visibility_state):
    return (gr.update(visible=not visibility_state), not visibility_state)


def update_start_date(start_date):
    return start_date


def update_end_date(end_date):  #
    return end_date


def reset_date_filters():
    default_start = None
    default_end = None
    return (default_start, default_end, gr.update(value=None), gr.update(value=None))


def set_today_dates():
    today = datetime.date.today()
    today_datetime = datetime.datetime.combine(today, datetime.datetime.min.time())
    return (
        today,
        today,
        gr.update(value=today_datetime),
        gr.update(value=today_datetime),
    )


def set_week_dates():
    today = datetime.date.today()
    today_datetime = datetime.datetime.combine(today, datetime.datetime.min.time())
    start_of_week = today - datetime.timedelta(days=7)
    start_of_week_datetime = datetime.datetime.combine(start_of_week, datetime.datetime.min.time())
    return (
        start_of_week,
        today,
        gr.update(value=start_of_week_datetime),
        gr.update(value=today_datetime),
    )


def set_two_week_dates():
    today = datetime.date.today()
    today_datetime = datetime.datetime.combine(today, datetime.datetime.min.time())
    start_of_two_week = today - datetime.timedelta(days=14)
    start_of_two_week_datetime = datetime.datetime.combine(start_of_two_week, datetime.datetime.min.time())
    return (
        start_of_two_week,
        today,
        gr.update(value=start_of_two_week_datetime),
        gr.update(value=today_datetime),
    )


def set_month_dates():
    today = datetime.date.today()
    today_datetime = datetime.datetime.combine(today, datetime.datetime.min.time())
    start_of_month = today - datetime.timedelta(days=30)
    start_of_month_datetime = datetime.datetime.combine(start_of_month, datetime.datetime.min.time())
    return (
        start_of_month,
        today,
        gr.update(value=start_of_month_datetime),
        gr.update(value=today_datetime),
    )


def set_half_year_dates():
    today = datetime.date.today()
    today_datetime = datetime.datetime.combine(today, datetime.datetime.min.time())
    start_of_half_year = today - datetime.timedelta(days=180)
    start_of_half_year_datetime = datetime.datetime.combine(start_of_half_year, datetime.datetime.min.time())
    return (
        start_of_half_year,
        today,
        gr.update(value=start_of_half_year_datetime),
        gr.update(value=today_datetime),
    )
