import pickle

model_file = '/workspaces/ml-zoomcamp-2025-homework/05-homework/pipeline_v1.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

to_score = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

X = dv.transform([to_score])
y_pred = model.predict_proba(X)[0, 1]

print('input', to_score)
print('converted probability', y_pred)

