from django.urls import path
from . import views
from django.urls import path
from basicConcepts.views import (
    ObtainAuthTokenView,
    docter_view,
   	patient_view,
    ChangePasswordView
)
from .views import *
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register("doctorRegistration",Doctor_ViewSet, basename="Doctor Registration")
router.register("PatientRegistration", Patient_ViewSet,basename="Patient Registration")


urlpatterns = [
    path('blood', views.bloodReqt, name='bloodReqt'),
    path('diabetes', views.diabetes_result, name='diabetes_result'),
    path('alzheimer', views.alzheimer_result, name='alzheimer_result'),
    path('heart', views.heart_disease_result, name='heart_disease_result'),
    path('parkinson', views.parkinson_result, name='parkinson_result'),
    path('posts/', views.PostView.as_view(), name='posts_list'),
    path('docter_register', docter_view, name="docter_register"),
    path('patient_register', patient_view, name="patient_register"),
    path('login', ObtainAuthTokenView.as_view(), name="login"),
    path('change_password/', ChangePasswordView.as_view(), name="change_password"),
    path('blood_data/<int:patient>', views.blooddata, name='blood_data'),
    path('diabtes_data/<int:patient>', views.diabtesdata, name='diabtes_data'),
    path('alzheimers_data/<int:patient>',views.alzheimersdata, name='alzheimers_data'),
    path('heart_data/<int:patient>',views.heartdata, name='heart_data'),
    path('parkinsons_data/<int:patient>',views.parkinsonsdata, name='parkinsons_data'),
    # path('chest_data/<int:patient>', views.chestdata, name='chest_data'),
    path('blood_data_all', views.blood_data_all, name='blood_data_all'),
    path('diabtes_data_all', views.diabtes_data_all, name='diabtes_data_all'),
    path('alzheimer_data_all', views.alzheimer_data_all, name='alzheimer_data_all'),
    path('heart_data_all', views.heart_data_all, name='heart_data_all'),
    path('parkinson_data_all', views.parkinson_data_all, name='parkinson_data_all'),
    
]

urlpatterns = urlpatterns+ router.urls