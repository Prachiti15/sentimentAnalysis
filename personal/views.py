from django.shortcuts import render
from .colab import utils
import json

# Create your views here.
output = []


def loading_screen(request):
    print(request.headers)
    return render(request, "personal/home.html", {})


def takeSecond(elem):
    return elem[1]


def home_screen_view(request):
    global output
    mylist = ["Happy", "Satisfied", "Dissapointed", "Mixed"]
    nutlabels = json.dumps(mylist)
    nutdata = json.dumps([12, 19, 3, 5])
    barlabels = json.dumps(mylist)
    bardata = json.dumps([220, 930, 310, 560])
    if request.method == 'POST':
        question = request.POST.get('keyword')
        output = utils.getUrl(question)
        return render(request, "personal/products.html", {'answer': output, 'requested': False})
    return render(request, "personal/index.html")


def detailed_view(request, id):
    mylist = ["Satisfied", "Happy", "Mixed", "Dissapointed"]
    satis = output[id][3].count(0)
    happy = output[id][3].count(1)
    mixed = output[id][3].count(2)
    diss = output[id][3].count(3)
    ml2 = ["Product 1",
           "Product 2", "Product 3", "This Product"]
    nutlabels = json.dumps(mylist)
    nutdata = json.dumps([happy, satis, mixed, diss])
    barlabels = json.dumps(ml2)
    bardata = json.dumps(
        [output[0][1], output[1][1], output[2][1], output[id][1]])
    return render(request, "personal/specificProduct.html", {'Product_Name': output[id][0], "score": output[id][1], "nutdata": nutdata, "nutlabels": nutlabels, "bardata": bardata, "barlabels": barlabels})

# OnePlus Nord CE 3 Lite 5G (Chromatic Gray, 8GB RAM, 128GB Storage) 4.35
# OnePlus Nord CE 3 Lite 5G (Pastel Lime, 8GB RAM, 128GB Storage) 4.35
# Oneplus Nord CE 3 5G (Grey Shimmer, 8GB RAM, 128GB Storage) 4.1
# OnePlus Nord 3 5G (Tempest Gray, 16GB RAM, 256GB Storage)
