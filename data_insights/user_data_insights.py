import pandas as pd
from enum import Enum

from utils.file_generation import write_df_to_csv

class Gender(Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    NOT_SPECIFIED = "NOT_MENTIONED"

def get_registered_users_on_date(users_df, date, gender=None):
    if gender is None:
        return len(users_df[users_df['registrationDate'] == date])

    return len(users_df[(users_df['registrationDate'] == date) & (users_df['gender'] == gender)])

def process_total_registered_users_on_date_and_store():
    users_df = pd.read_csv('./data/processed_user_data.csv')

    total_users_by_date_df = pd.DataFrame(columns=['date', 'number_of_users'])
    
    total_users_on_curr_date = 0

    for index, row in users_df.iterrows():
        total_users_on_curr_date += get_registered_users_on_date(users_df, row['registrationDate'])
        total_users_by_date_df = pd.concat([total_users_by_date_df, pd.DataFrame({'date': [row['registrationDate']], 'number_of_users': [total_users_on_curr_date]})])

    write_df_to_csv(total_users_by_date_df, './data/total_registered_users_by_day.csv')

def process_total_gendered_registered_users_on_date_and_store():
    users_df = pd.read_csv('./data/processed_user_data.csv')

    total_male_users_by_date_df = pd.DataFrame(columns=['date', 'number_of_users'])
    total_female_users_by_date_df = pd.DataFrame(columns=['date', 'number_of_users'])
    total_ungendered_users_by_date_df = pd.DataFrame(columns=['date', 'number_of_users'])

    total_male_users_on_curr_date = 0
    total_female_users_on_curr_date = 0
    total_ungendered_users_on_curr_date = 0

    for index, row in users_df.iterrows():
        total_male_users_on_curr_date += get_registered_users_on_date(users_df=users_df, date=row['registrationDate'], gender=Gender.MALE.value)
        total_male_users_by_date_df = pd.concat([total_male_users_by_date_df, pd.DataFrame({'date': [row['registrationDate']], 'number_of_users': [total_male_users_on_curr_date]})])

        total_female_users_on_curr_date += get_registered_users_on_date(users_df=users_df, date=row['registrationDate'], gender=Gender.FEMALE.value)
        total_female_users_by_date_df = pd.concat([total_female_users_by_date_df, pd.DataFrame({'date': [row['registrationDate']], 'number_of_users': [total_female_users_on_curr_date]})])

        total_ungendered_users_on_curr_date += get_registered_users_on_date(users_df=users_df, date=row['registrationDate'], gender=Gender.NOT_SPECIFIED.value)
        total_ungendered_users_by_date_df = pd.concat([total_ungendered_users_by_date_df, pd.DataFrame({'date': [row['registrationDate']], 'number_of_users': [total_ungendered_users_on_curr_date]})])

    write_df_to_csv(total_male_users_by_date_df, './data/total_male_registered_users_by_day.csv')
    write_df_to_csv(total_female_users_by_date_df, './data/total_female_registered_users_by_day.csv')
    write_df_to_csv(total_ungendered_users_by_date_df, './data/total_ungendered_registered_users_by_day.csv')
