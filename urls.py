from django.urls import path
# from .views import index, run_script
from . import views
urlpatterns = [
    path('', views.index, name='index'),  # Home page
    # path('run-script/', run_script, name='run_script'),  # Endpoint to run script
    # path("simple_function",views.simple_function)
    path("external/",views.external,name="script")
]
