def next_birthday(date, birthdays):
    month, day = date

    dates = list(birthdays.keys())

    future_dates = []
    for d in dates:
        if d > (month, day):
            future_dates.append(d)

    if future_dates:
        next_date = min(future_dates)
    else:
        next_date = min(dates)

    return next_date, birthdays[next_date]
