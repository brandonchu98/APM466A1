import pandas as pd
import math
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import linalg

par_value = 100
today_date = '2/1/2021'
bonds_df = pd.read_csv("bond_data.csv")

usable_bonds = ["CA135087K296", "CA135087K601", "CA135087L286", "CA135087L773", "CA135087A610", "CA135087J546",
                "CA135087J967", "CA135087K528", "CA135087K940", "CA135087L518"]

# filter the date frame for usable bonds
filtered_bonds_df = bonds_df[bonds_df.ISIN.isin(usable_bonds)]

# reorder filtered_bonds_df by maturity dates
sorted_dates = sorted(filtered_bonds_df['Maturity Date'],
                      key=lambda date: datetime.strptime(date, "%m/%d/%Y"),
                      reverse=False)
filtered_bonds_df = filtered_bonds_df.set_index('Maturity Date').loc[sorted_dates].reset_index()

# convert coupon column from percentages to decimals
filtered_bonds_df['Coupon'] = filtered_bonds_df['Coupon'].str[:-1]
filtered_bonds_df['Coupon'] = pd.to_numeric(filtered_bonds_df['Coupon'], downcast="float") / 100


def date_difference(date1, date2):
    date1 = datetime.strptime(date1, "%m/%d/%Y")
    date2 = datetime.strptime(date2, "%m/%d/%Y")
    return abs((date1.year - date2.year) * 12 + (date1.month - date2.month)) / 12


def sum_previous_terms(index, spot_rates):
    count = 0
    if index >= 6:
        for i in range(index+1):
            count += (1 + (spot_rates[i] / 2)) ** (
                    (-date_difference(today_date, filtered_bonds_df.at[index, 'Maturity Date']) - (0.5 * (i + 1))) * 2)
    else:
        for i in range(index):
            count += (1 + (spot_rates[i] / 2)) ** (
                    (-date_difference(today_date, filtered_bonds_df.at[index, 'Maturity Date']) - (0.5 * (i + 1))) * 2)
    if date_difference(today_date, filtered_bonds_df.at[index, 'Maturity Date']) % 0.5 == 0:
        return count + 1
    else:
        return count


def zero_coupon_yield(price, coupon_rate, time_to_maturity):
    return (((100 + (coupon_rate * 50)) / price) ** (1 / (time_to_maturity * 2)) - 1) * 2


def get_accrued_interest(coupon_rate, maturity_date, date):
    last_payment_time = date_difference(date, maturity_date)
    while last_payment_time >= 0:
        last_payment_time -= 0.5
    return (coupon_rate * 100) * abs(last_payment_time)


def interpolate(x, x1, y1, x2, y2):
    return y1 + ((x - x1) * ((y2 - y1) / (x2 - x1)))


def interpolation_eqn(r_4, r_5, cp):
    fn = (1 + ((r_4 + (0.25 * ((r_5 - r_4) / 0.75))) / 2)) ** 6
    gn = (cp / fn) + ((cp + 100) / ((1 + (r_5 / 2)) ** 7))
    return gn


def get_spot_rates(date):
    spot_rates = []
    spot_dict = {0.0: 0.0}
    for i in range(5):
        cp = filtered_bonds_df.at[i, 'Coupon'] * 50
        if date_difference(filtered_bonds_df.at[i, 'Maturity Date'], today_date) % 0.5 == 0:
            rate = zero_coupon_yield(filtered_bonds_df.at[i, date] +
                                     get_accrued_interest(filtered_bonds_df.at[i, 'Coupon'],
                                                          filtered_bonds_df.at[i, 'Maturity Date'], today_date) -
                                     (sum_previous_terms(i, spot_rates) * cp),
                                     filtered_bonds_df.at[i, 'Coupon'],
                                     date_difference(today_date, filtered_bonds_df.at[i, 'Maturity Date']))
            spot_rates.append(rate)
            spot_dict[date_difference(today_date, filtered_bonds_df.at[i, 'Maturity Date'])] = spot_rates[i]
        elif date_difference(filtered_bonds_df.at[i, 'Maturity Date'], today_date) % 0.5 != 0:
            num_payments = date_difference(today_date, filtered_bonds_df.at[i, 'Maturity Date']) // 0.5
            interpolated_spots = []
            for j in range(i):
                interpolated_spots.append(
                    interpolate(date_difference(today_date, filtered_bonds_df.at[i, 'Maturity Date']) - (0.5 * (i - j)),
                                list(spot_dict)[j],
                                spot_dict.get(list(spot_dict)[j]),
                                list(spot_dict)[j + 1],
                                spot_dict.get(list(spot_dict)[j + 1])))
            rate = zero_coupon_yield(filtered_bonds_df.at[i, date] +
                                     get_accrued_interest(filtered_bonds_df.at[i, 'Coupon'],
                                                          filtered_bonds_df.at[i, 'Maturity Date'], today_date) -
                                     (sum_previous_terms(int(num_payments), interpolated_spots) * cp),
                                     filtered_bonds_df.at[i, 'Coupon'],
                                     date_difference(today_date, filtered_bonds_df.at[i, 'Maturity Date']))
            spot_dict[date_difference(today_date, filtered_bonds_df.at[i, 'Maturity Date'])] = rate

    interpolated_spots = []
    cp = filtered_bonds_df.at[5, 'Coupon'] * 50
    for j in range(5):
        interpolated_spots.append(
            interpolate(date_difference(today_date, filtered_bonds_df.at[5, 'Maturity Date']) - 0.5 - (0.5 * (5 - j)),
                        list(spot_dict)[j],
                        spot_dict.get(list(spot_dict)[j]),
                        list(spot_dict)[j + 1],
                        spot_dict.get(list(spot_dict)[j + 1])))
    lhs = filtered_bonds_df.at[5, date] + get_accrued_interest(filtered_bonds_df.at[5, 'Coupon'],
                                                               filtered_bonds_df.at[5, 'Maturity Date'],
                                                               today_date) - \
          (sum_previous_terms(5, interpolated_spots) * cp)

    r_5 = 0
    r_4 = list(spot_dict.values())[5]

    while round(interpolation_eqn(r_4, r_5, cp), 2) != round(lhs, 2):
        r_5 += 0.0000001

    spot_dict[date_difference(today_date, filtered_bonds_df.at[5, 'Maturity Date'])] = r_5

    spot_dict[2.5] = interpolate(2.5,
                                 date_difference(today_date, filtered_bonds_df.at[4, 'Maturity Date']),
                                 r_4,
                                 date_difference(today_date, filtered_bonds_df.at[5, 'Maturity Date']),
                                 r_5)

    spot_dict[3.0] = interpolate(3,
                               2.5,
                               spot_dict[2.5],
                               date_difference(today_date, filtered_bonds_df.at[5, 'Maturity Date']),
                               r_5)

    spot_rates.append(spot_dict[2.5])
    spot_rates.append(spot_dict[3])

    interpolated_spots.append(spot_dict[date_difference(today_date, filtered_bonds_df.at[4, 'Maturity Date'])])
    interpolated_spots.append(r_5)
    for i in range(6, 10):
        cp = filtered_bonds_df.at[i, 'Coupon'] * 50
        rate = zero_coupon_yield(filtered_bonds_df.at[i, date] +
                                 get_accrued_interest(filtered_bonds_df.at[i, 'Coupon'],
                                                      filtered_bonds_df.at[i, 'Maturity Date'], today_date) -
                                 (sum_previous_terms(i, interpolated_spots) * cp),
                                 filtered_bonds_df.at[i, 'Coupon'],
                                 date_difference(today_date, filtered_bonds_df.at[i, 'Maturity Date']))
        interpolated_spots.append(rate)
        spot_dict[date_difference(today_date, filtered_bonds_df.at[i, 'Maturity Date'])] = rate

        x = interpolate((i + 1) * 0.5,
                        date_difference(today_date, filtered_bonds_df.at[i - 1, 'Maturity Date']),
                        interpolated_spots[i - 1],
                        date_difference(today_date, filtered_bonds_df.at[i, 'Maturity Date']),
                        interpolated_spots[i])

        spot_dict[(i + 1) * 0.5] = x

    final_rates = []
    for i in range(len(spot_dict)):
        if list(spot_dict)[i] % 0.5 == 0:
            final_rates.append(spot_dict[list(spot_dict)[i]])

    return final_rates[1:]


def interpolate_bonds(date):
    interpolated_spot_rates = []
    for i in range(10):
        term = (i + 1) * 0.5
        j = 0
        while date_difference(date, filtered_bonds_df.at[j, 'Maturity Date']) >= term or \
                date_difference(date, filtered_bonds_df.at[j + 1, 'Maturity Date']) <= term:
            j += 1
        y1 = get_spot_rates(date)[j]
        y2 = get_spot_rates(date)[j + 1]
        x1 = date_difference(date, filtered_bonds_df.at[j, 'Maturity Date'])
        x2 = date_difference(date, filtered_bonds_df.at[j + 1, 'Maturity Date'])
        interpolation = y1 + ((term - x1) * ((y2 - y1) / (x2 - x1)))
        interpolated_spot_rates.append(interpolation)
    return interpolated_spot_rates


def get_forward_rates(date):
    spots = get_spot_rates(date)
    forwards = []
    for j in range(4):
        index = 2 * (j + 1) + 1
        forwards.append((((1 + spots[index]) ** (j + 2)) / (1 + (spots[1]))) - 1)
    return forwards


def get_maturity_dates(date):
    dates = []
    for i in range(filtered_bonds_df.shape[0]):
        dates.append(date_difference(today_date, filtered_bonds_df.at[i, 'Maturity Date']))
    return dates


plot1 = plt.figure(1)
terms = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
plt.plot(terms, get_spot_rates('1/18/2021'))
plt.plot(terms, get_spot_rates('1/19/2021'))
plt.plot(terms, get_spot_rates('1/20/2021'))
plt.plot(terms, get_spot_rates('1/21/2021'))
plt.plot(terms, get_spot_rates('1/22/2021'))
plt.plot(terms, get_spot_rates('1/25/2021'))
plt.plot(terms, get_spot_rates('1/26/2021'))
plt.plot(terms, get_spot_rates('1/27/2021'))
plt.plot(terms, get_spot_rates('1/28/2021'))
plt.plot(terms, get_spot_rates('1/29/2021'))


plt.title('Spot Curve')
plt.xlabel('Term')
plt.ylabel('Spot Rates')


plot2 = plt.figure(2)
forward_terms = [2, 3, 4, 5]
plt.plot(forward_terms, get_forward_rates('1/18/2021'))
plt.plot(forward_terms, get_forward_rates('1/19/2021'))
plt.plot(forward_terms, get_forward_rates('1/20/2021'))
plt.plot(forward_terms, get_forward_rates('1/21/2021'))
plt.plot(forward_terms, get_forward_rates('1/25/2021'))
plt.plot(forward_terms, get_forward_rates('1/26/2021'))
plt.plot(forward_terms, get_forward_rates('1/27/2021'))
plt.plot(forward_terms, get_forward_rates('1/28/2021'))
plt.plot(forward_terms, get_forward_rates('1/29/2021'))


plt.title('Forward Curve')
plt.xlabel('Term')
plt.ylabel('Forward Rates')


def current_yield(price, coupon_rate):
    return coupon_rate * 100 / price


def ytm_formula(ytm, coupon_payment, time_to_maturity):
    number_of_payments = time_to_maturity // 0.5
    count = 0
    for i in range(int(number_of_payments)):
        count += coupon_payment / ((1 + (ytm / 2)) ** ((time_to_maturity - (0.5 * (i + 1))) * 2))
    return count + ((par_value + coupon_payment) / ((1 + ytm) ** time_to_maturity))


def get_ytm(bond, date):
    ytm = -.1
    cp = bond.at['Coupon'] * 50

    if date_difference(today_date, bond.at['Maturity Date']) % 0.5 == 0:
        price = bond.at[date] + get_accrued_interest(bond.at['Coupon'], bond.at['Maturity Date'], today_date)
    else:
        price = bond.at[date] + get_accrued_interest(bond.at['Coupon'], bond.at['Maturity Date'], today_date)

    while round(price, 2) != \
            round(ytm_formula(ytm, cp, date_difference(today_date, bond.at['Maturity Date'])), 2):
        ytm += 0.00001
    return ytm


def get_all_yields(date):
    yields = []
    for i in range(10):
        yields.append(get_ytm(filtered_bonds_df.iloc[i], date))
    return yields


def interpolate_ytm(date):
    dates = get_maturity_dates(date)
    ytm_list = get_all_yields(date)
    interpolated_ytm = []
    for i in range(len(dates)):
        term = (i + 1) * 0.5
        if dates[i] % 0.5 != 0:
            j = 0
            while date_difference(today_date, filtered_bonds_df.at[j, 'Maturity Date']) >= term or \
                         date_difference(today_date, filtered_bonds_df.at[j + 1, 'Maturity Date']) <= term:
                j += 1
            interpolated_ytm.append(interpolate((i + 1) * 0.5,
                                                dates[j],
                                                ytm_list[j],
                                                dates[j + 1],
                                                ytm_list[j + 1]))
        else:
            interpolated_ytm.append(ytm_list[i])
    return interpolated_ytm


interpolated_ytm_days = [interpolate_ytm('1/18/2021'), interpolate_ytm('1/19/2021'), interpolate_ytm('1/20/2021'),
                         interpolate_ytm('1/21/2021'), interpolate_ytm('1/22/2021'), interpolate_ytm('1/25/2021'),
                         interpolate_ytm('1/26/2021'), interpolate_ytm('1/27/2021'), interpolate_ytm('1/28/2021'),
                         interpolate_ytm('1/29/2021')]

all_forwards = [get_forward_rates('1/18/2021'), get_forward_rates('1/19/2021'), get_forward_rates('1/20/2021'),
                get_forward_rates('1/21/2021'), get_forward_rates('1/22/2021'), get_forward_rates('1/25/2021'),
                get_forward_rates('1/26/2021'), get_forward_rates('1/27/2021'), get_forward_rates('1/28/2021'),
                get_forward_rates('1/29/2021')]


def ytm_year_log_returns(year):
    index = (2 * year) - 1
    x = []
    for i in range(len(interpolated_ytm_days) - 1):
        if interpolated_ytm_days[i][index] < 0:
            interpolated_ytm_days[i][index] = 0.0001
        elif interpolated_ytm_days[i + 1][index] < 0:
            interpolated_ytm_days[i + 1][index] = 0.0001
        x.append(math.log(interpolated_ytm_days[i + 1][index] / interpolated_ytm_days[i][index]))
    return x


def forward_log_returns(year):
    index = year - 1
    x = []
    for i in range(len(all_forwards) - 1):
        x.append(math.log(all_forwards[i + 1][index] / all_forwards[i][index]))
    return x

time_series_data = {"x1": [], "x2": [], 'x3': [], 'x4': [], 'x5': []}
for i in range(5):
    time_series_data[list(time_series_data)[i]] = ytm_year_log_returns(i + 1)

yield_df = pd.DataFrame(time_series_data, columns=['x1', 'x2', 'x3', 'x4', 'x5'])


time_series_forward = {"x1": [], "x2": [], 'x3': [], 'x4': []}
for j in range(4):
    time_series_forward[list(time_series_forward)[j]] = forward_log_returns(j + 1)

forward_df = pd.DataFrame(time_series_forward, columns=['x1', 'x2', 'x3', 'x4'])

print(yield_df)
print(yield_df.cov())

print(forward_df)
print(forward_df.cov())

print(linalg.eig(yield_df.cov()))
print(linalg.eig(forward_df.cov()))


plot3 = plt.figure(3)
plt.plot(terms, interpolate_ytm('1/18/2021'))
plt.plot(terms, interpolate_ytm('1/19/2021'))
plt.plot(terms, interpolate_ytm('1/20/2021'))
plt.plot(terms, interpolate_ytm('1/21/2021'))
plt.plot(terms, interpolate_ytm('1/22/2021'))
plt.plot(terms, interpolate_ytm('1/25/2021'))
plt.plot(terms, interpolate_ytm('1/26/2021'))
plt.plot(terms, interpolate_ytm('1/27/2021'))
plt.plot(terms, interpolate_ytm('1/28/2021'))
plt.plot(terms, interpolate_ytm('1/29/2021'))


plt.title('Yield Curve')
plt.xlabel('Term')
plt.ylabel('Yields')

plt.show()
