import requests as requests
import base64

with open('/home/esteban/Escritorio/perrito.jpeg', 'rb') as f:
    im_b64 = base64.b64encode(f.read())

payload = {'id': '123', 'type': 'jpg', 'box': [0, 0, 100, 100], 'image': im_b64}
print(payload)
url = 'http://192.168.1.111:5002/im_size'
# r = requests.post(url, data=payload)
# print(r.json())


# import requests as requests
# import base64
#
# id_img =  '37NGE_03_20180121'
# payload = {'id': '123', 'type': 'jpg', 'box': [0, 0, 100, 100], 'image': id_img}
# url = 'http://192.168.1.111:5002/im_size'
# r = requests.post(url, data=payload)
# print(r.json())