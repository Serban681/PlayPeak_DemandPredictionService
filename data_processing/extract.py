def process_user_data(user_data):
    get_user_essentials = lambda user: {
        'id': user['id'],
        'registrationDate': user['registrationDate'],
        'age': user['age'],
        'gender': user['gender']
    }

    filtered_users = filter(lambda user: user['role'] != 'ADMIN', user_data)

    processed_user_data = list(map(get_user_essentials, filtered_users))
    
    return processed_user_data

def process_order_data(order_data, product_variance_id):
    def get_order_essentials(order):
        matching_entry = next(
            (entry for entry in order['cart']['cartEntries']
             if entry['productVariance']['id'] == product_variance_id),
            None
        )
        
        if matching_entry is None:
            return None
        
        return {
            'id': order['id'],
            'userId': order['user']['id'],
            'productVarianceId': matching_entry['productVariance']['id'],
            'quantity': matching_entry['quantity'],
            'orderDate': order['orderDate']
        }

    processed_order_data = list(filter(None, map(get_order_essentials, order_data)))

    return processed_order_data
