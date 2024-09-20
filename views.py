from django.http import HttpResponse
from django.shortcuts import render

def home(request):
    return HttpResponse("Welcome to s2s")
def homePage(request):
    return render(request,"index.html")
