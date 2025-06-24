def process_orders_by_date(order_data):
    order_dates = [order['orderDate'] for order in order_data]
    
    from collections import Counter
    orders_per_day_counter = Counter(order_dates)
    
    total_orders_by_date = [
        {'date': date, 'number_of_orders': count}
        for date, count in orders_per_day_counter.items()
    ]
    
    return total_orders_by_date

def process_total_users_by_date(user_data):
    registration_dates = [user['registrationDate'] for user in user_data]
    
    from collections import Counter
    users_per_day_counter = Counter(registration_dates)
    
    all_dates_sorted = sorted(users_per_day_counter.keys())
    
    cumulative_total = 0
    total_users_by_date = []
    
    for date in all_dates_sorted:
        daily_count = users_per_day_counter[date]
        cumulative_total += daily_count
        total_users_by_date.append({
            'date': date,
            'number_of_users': cumulative_total
        })
    
    return total_users_by_date
