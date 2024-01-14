from django.urls import path
from . import views


urlpatterns = [
    path('', views.pneumonia, name='pneumonia'),
    path('predict_pneumonia/', views.predict_pneumonia, name='predict_pneumonia'),
]