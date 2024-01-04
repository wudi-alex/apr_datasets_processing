import json

file_name = 'codellama_7b_fillme.json'

with open('input_base.json', 'r') as file:
    data = json.load(file)

for key in data['data']:
    data['data'][key]['input'] = data['data'][key]['pre'] + "<FILL_ME>" + data['data'][key]['post']

json.dump(data, open(file_name, 'w'), indent=2)

with open(file_name, 'r') as file:
    data1 = json.load(file)['data']
    for i in data1:
        print(f'----------{i}----------')
        print(i)
        print(data1[i]['input'])
