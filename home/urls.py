from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('face_detection/', views.face_detection, name='face_detection'),
    path('face_detection/detection_face', views.detection_face, name='detection_face'),

    path('face_mask_detection/', views.face_mask_detection, name='face_mask_detection'),
    path('face_mask_detection/predictImage', views.predictImage, name='predictImage'),
    
    path('dataset/', views.dataset, name='dataset'),
    path('accouracy/', views.accouracy, name='accouracy'),
    path('about/', views.about, name='about')
]