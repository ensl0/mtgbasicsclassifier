import json
import requests
from PIL import Image
from model import Basics_Model

def save_all_lands():
    f = open('C://Datasets//unique-artwork.json', encoding="utf8")
    d = json.load(f)
    good_types = ['Land', 'Basic']
    for i, row in enumerate(d):
        try:
            types = row['type_line'].split()
            if set(types).intersection(good_types) != set(good_types):
                continue
            land_type = types[-1]
            if land_type == 'Land':
                land_type = 'Wastes'
            r = requests.get(row['image_uris']['art_crop'], stream=True)
            image = Image.open(r.raw)
            image.save('S://MTGARTLANDS//' + land_type.upper() + '//' + row['illustration_id'] + '.jpg')
        except:
            continue

m = Basics_Model()
m.Init_Model()
m.test_Model()
#m.predict_model('https://i.pinimg.com/originals/50/a3/84/50a38451cf5a2b4a68f737abf10e7887.jpg')