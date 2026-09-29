from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from src.serving.inference import predict
import gradio as gr

app=FastAPI()

# Root Page string (just to check whether everything is fine or not)
@app.get("/")
def root():
    return {'status':'ok'}


# Model prediction api

# Defining a schema for user input of all required data

class user_input(BaseModel):
    age: int = Field(ge=1, le=100)
    days_since_last_login: float = Field(ge=0.0)
    avg_time_spent_in_seconds: float = Field(ge=0.0)
    avg_transaction_value: float = Field(ge=0.0)
    avg_frequency_login_days: float = Field(ge=0.0)
    points_in_wallet: float = Field(ge=0.0)
    joining_date: str = Field(default="2015-01-01", description="Joining date in YYYY-MM-DD format")
    gender: Literal['M', 'F', 'Unknown']
    region_category: Literal['Village', 'Town', 'City', 'Unknown']
    membership_category: Literal['Basic Membership', 'No Membership', 'Gold Membership', 'Platinum Membership', 'Premium Membership', 'Silver Membership']
    joined_through_referral: Literal['Yes', 'No', '?']
    preferred_offer_types: Literal['Gift Vouchers/Coupons','Without Offers','Credit/Debit Card Offers']
    medium_of_operation: Literal['Desktop', 'Smartphone', 'Both', '?']
    internet_option: Literal['Mobile_Data', 'Wi-Fi', 'Fiber_Optic']
    used_special_discount: Literal['Yes', 'No']
    offer_application_preference: Literal['Yes', 'No']
    past_complaint: Literal['Yes', 'No']
    complaint_status: Literal['Not Applicable', 'Solved', 'Solved in Follow-up', 'Unsolved', 'No Information Available']
    feedback: Literal['Poor Website','Poor Product Quality','Poor Customer Service','No reason specified','Reasonable Price','Too many ads','Products always in Stock','User Friendly Website','Quality Customer Care']



@app.post("/predict")
def api_predict(data:user_input):
    response=predict(data)
    return response

############################ GRADIO #######################################

def ui_predict(age, days_since_last_login, avg_time_spent_in_seconds, avg_transaction_value, avg_frequency_login_days, points_in_wallet, joining_date, gender, region_category, membership_category, joined_through_referral, preferred_offer_types, medium_of_operation, internet_option, used_special_discount, offer_application_preference, past_complaint, complaint_status, feedback):
    
    ui_user_input_dict={'age':age, 
                        'days_since_last_login':days_since_last_login, 
                        'avg_time_spent_in_seconds':avg_time_spent_in_seconds, 
                        'avg_transaction_value':avg_transaction_value, 
                        'avg_frequency_login_days':avg_frequency_login_days, 
                        'points_in_wallet':points_in_wallet, 
                        'joining_date':joining_date, 
                        'gender':gender, 
                        'region_category':region_category, 
                        'membership_category':membership_category, 
                        'joined_through_referral':joined_through_referral, 
                        'preferred_offer_types':preferred_offer_types, 
                        'medium_of_operation':medium_of_operation, 
                        'internet_option':internet_option, 
                        'used_special_discount':used_special_discount, 
                        'offer_application_preference':offer_application_preference, 
                        'past_complaint':past_complaint, 
                        'complaint_status':complaint_status, 
                        'feedback':feedback}

    return predict(ui_user_input_dict)


interface=gr.Interface(fn=ui_predict,
                       inputs=[gr.Number(label='Age', minimum=1, maximum=100),
                               gr.Number(label='Days since last login', minimum=0),
                               gr.Number(label='Average transaction value', minimum=0),
                               gr.Number(label='Average time spent (seconds)', minimum=0),
                               gr.Number(label='Average login frequency (days)', minimum=0),
                               gr.Number(label='Points in wallet', minimum=0),
                               gr.DateTime(label='Joining date', type="string", value="2015-01-01", include_time=False),
                               gr.Dropdown(label='Gender', choices=['M', 'F', 'Unknown']),
                               gr.Dropdown(label='Region', choices=['Village', 'Town', 'City', 'Unknown']),
                               gr.Dropdown(label='Membership category', choices=['Basic Membership', 'No Membership', 'Gold Membership', 'Platinum Membership', 'Premium Membership', 'Silver Membership']),
                               gr.Dropdown(label='Joined via referral', choices=['Yes', 'No', '?']),
                               gr.Dropdown(label='Preferred offer type', choices=['Gift Vouchers/Coupons','Without Offers','Credit/Debit Card Offers']),
                               gr.Dropdown(label='Device used for operation', choices=['Desktop', 'Smartphone', 'Both', '?']),
                               gr.Dropdown(label='Internet connection type', choices=['Mobile_Data', 'Wi-Fi', 'Fiber_Optic']),
                               gr.Dropdown(label='Used special discount', choices=['Yes', 'No']),
                               gr.Dropdown(label='Offer application preference', choices=['Yes', 'No']),
                               gr.Dropdown(label='Had a past complaint', choices=['Yes', 'No']),
                               gr.Dropdown(label='Complaint status', choices=['Not Applicable', 'Solved', 'Solved in Follow-up', 'Unsolved', 'No Information Available']),
                               gr.Dropdown(label='Feedback category', choices=['Poor Website','Poor Product Quality','Poor Customer Service','No reason specified','Reasonable Price','Too many ads','Products always in Stock','User Friendly Website','Quality Customer Care'])],
                        outputs=gr.Number(label='Churn Score'),
                        title='User Churn Score Predictor',
                        description='Predict the Risk Score of Customer',
                        theme=gr.themes.Soft())


app=gr.mount_gradio_app(app, interface, path="/ui")


if __name__=='__main__':

    ui_prediction_feature_fields = [
        "age",                                 #
        "days_since_last_login",               #
        "avg_time_spent_in_seconds",           #
        "avg_transaction_value",               #
        "avg_frequency_login_days",            #
        "points_in_wallet",                    #
        "joining_date",
        "gender",                               #
        "region_category",                       #
        "membership_category",                   #
        "joined_through_referral",               #
        "preferred_offer_types",                #
        "medium_of_operation",                   #
        "internet_option",                      #
        "used_special_discount",                #
        "offer_application_preference",        #
        "past_complaint",                      # 
        "complaint_status",                    #
        "feedback",                             #
    ]


    input_dict=dict.fromkeys(ui_prediction_feature_fields)

    input_dict["age"]=43
    input_dict["days_since_last_login"]=56
    input_dict["avg_time_spent_in_seconds"]=134
    input_dict["avg_transaction_value"]=45
    input_dict["avg_frequency_login_days"]=6
    input_dict["points_in_wallet"]=45
    input_dict["joining_date"]='2015-04-19'
    input_dict["gender"]='M'
    input_dict["region_category"]='Village'
    input_dict["membership_category"]='Silver Membership'
    input_dict["joined_through_referral"]='Yes'
    input_dict["preferred_offer_types"]='Without Offers'
    input_dict["medium_of_operation"]='Smartphone'
    input_dict["internet_option"]='Mobile_Data'
    input_dict["used_special_discount"]='Yes'
    input_dict["offer_application_preference"]='Yes'
    input_dict["past_complaint"]='No'
    input_dict["complaint_status"]='Not Applicable'
    input_dict["feedback"]='Poor Website'

    print(api_predict(input_dict))


