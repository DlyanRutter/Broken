import requests
import json

url = "enter heroku web app url here"

sample =  { 'age':50,
            'workclass':"Private", 
            'fnlgt':234721,
            'education':"Doctorate",
            'education_num':16,
            'marital_status':"Separated",
            'occupation':"Exec-managerial",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Female",
            'capital_gain':0,
            'capital_loss':0,
            'hours_per_week':50,
            'native_country':"United-States"
            }

data = json.dumps(sample)
response = requests.post(url, data=data )
print("response status code", response.status_code)
print("response content:")
print(response.json())

 