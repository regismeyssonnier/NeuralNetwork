from im import *
from mix import *


def write_img():
    t, n = load_image_one_filter_rand("2.jpg", 1)

    s="{"
    for i in t[0]:
        s+= str((1.0-i)*1000+1) + ","

    s+="}"
    print(s)


write_img()