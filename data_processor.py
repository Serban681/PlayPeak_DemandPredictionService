import pandas as pd

def process_and_store_user_data(user_data):
    get_user_essentials = lambda user: {
        'id': user['id'],
        'registrationDate': user['registrationDate'],
        'age': user['age'],
        'gender': user['gender']
    }

    processed_user_data = list(map(get_user_essentials, user_data))
    
    df = pd.DataFrame(processed_user_data)
    
    df.to_csv("./data/processed_user_data.csv", index=False)

def process_and_store_order_data(order_data):
    get_order_essentials = lambda order: {
        'id': order['id'],
        'userId': order['user']['id'],
        'productVarianceId': order['cart']['cartEntries'][0]['productVariance']['product']['id'],
        'quantity': order['cart']['cartEntries'][0]['quantity'],
        'orderDate': order['orderDate']
    }

    processed_order_data = list(map(get_order_essentials, order_data))
    
    df = pd.DataFrame(processed_order_data)

    df.to_csv("./data/processed_order_data.csv", index=False)


def process_order_data(order_data):
    get_order_essentials = lambda order: {
        'id': order['id'],
        'userId': order['user']['id'],
        'productVarianceId': order['cart']['cartEntries'][0]['productVariance']['product']['id'],
        'quantity': order['cart']['cartEntries'][0]['quantity'],
        'orderDate': order['orderDate']
    }

    processed_order_data = list(map(get_order_essentials, order_data))

    return processed_order_data


