from .Serializer import *
from .models import Account
from .models import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.generics import UpdateAPIView
from rest_framework.decorators import (api_view, authentication_classes,
                                       permission_classes)
from rest_framework.authtoken.models import Token
from rest_framework.authentication import TokenAuthentication
from rest_framework import status
from django.contrib.auth import authenticate
from basicConcepts.Serializer import (ChangePasswordSerializer,
                                      RegistrationSerializer)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import joblib
import cv2
from rest_framework.parsers import MultiPartParser, FormParser
import os
import pickle
from django.apps import apps
# from django import apps
from werkzeug.utils import secure_filename

from rest_framework import viewsets
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import status, filters



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# login


@permission_classes([])
class Doctor_ViewSet(viewsets.ModelViewSet):
    queryset = Account.objects.all()
    serializer_class = doctor_serializer 
    filter_backends = [DjangoFilterBackend,filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['id', 'email', 'username','is_docter']
    filterset_fields = ['id', 'email', 'username','is_docter']
    
@permission_classes([])
class Patient_ViewSet(viewsets.ModelViewSet):
    queryset = Account.objects.all()
    serializer_class = pation_serializer
    filter_backends = [DjangoFilterBackend,filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['id', 'email', 'username','is_docter']
    filterset_fields = ['id', 'email', 'username','is_docter']
    


@api_view(['POST', ])
@permission_classes([])
@authentication_classes([])
def docter_view(request):

    if request.method == 'POST':
        data = {}
        email = request.data.get('email', '0').lower()
        if validate_email(email) != None:
            data['error_message'] = 'That email is already in use.'
            data['response'] = 'Error'
            return Response(data)

        username = request.data.get('username', '0')
        if validate_username(username) != None:
            data['error_message'] = 'That username is already in use.'
            data['response'] = 'Error'
            return Response(data)

        serializer = RegistrationSerializer(
            data={**request.data, 'is_doctor': True})

        if serializer.is_valid():
            account = serializer.save()
            data['response'] = 'successfully registered new user.'
            data['email'] = account.email
            data['username'] = account.username
            data['pk'] = account.pk
            token = Token.objects.get(user=account).key
            data['token'] = token
        else:
            data = serializer.errors
        return Response(data)
###########################################################


@api_view(['POST', ])
@permission_classes([])
@authentication_classes([])
def patient_view(request):

    if request.method == 'POST':
        data = {}
        email = request.data.get('email', '0').lower()
        if validate_email(email) != None:
            data['error_message'] = 'That email is already in use.'
            data['response'] = 'Error'
            return Response(data)

        username = request.data.get('username', '0')
        if validate_username(username) != None:
            data['error_message'] = 'That username is already in use.'
            data['response'] = 'Error'
            return Response(data)

        serializer = RegistrationSerializer(
            data={**request.data, 'is_patient': True})

        if serializer.is_valid():
            account = serializer.save()
            data['response'] = 'successfully registered new user.'
            data['email'] = account.email
            data['username'] = account.username
            data['pk'] = account.pk
            token = Token.objects.get(user=account).key
            data['token'] = token
        else:
            data = serializer.errors
        return Response(data)


def validate_email(email):
    account = None
    try:
        account = Account.objects.get(email=email)
    except Account.DoesNotExist:
        return None
    if account != None:
        return email


def validate_username(username):
    account = None
    try:
        account = Account.objects.get(username=username)
    except Account.DoesNotExist:
        return None
    if account != None:
        return username
##################################################################


class ObtainAuthTokenView(APIView):

    authentication_classes = []
    permission_classes = []

    def post(self, request):
        context = {}

        email = request.data.get('email')
        password = request.data.get('password')
        account = authenticate(email=email, password=password)
        if account:
            try:
                token = Token.objects.get(user=account)
            except Token.DoesNotExist:
                token = Token.objects.create(user=account)
                
            is_doctor = Account.objects.get(email=email).is_docter
            context['response'] = 'Successfully authenticated.'
            context['pk'] = account.pk
            context['email'] = email.lower()
            context['token'] = token.key
            context['is_doctor'] = is_doctor
            return Response(context, status=status.HTTP_200_OK)

        else:
            context['response'] = 'Error'
            context['error_message'] = 'Invalid credentials'
            return Response(context , status=status.HTTP_401_UNAUTHORIZED)


class ChangePasswordView(UpdateAPIView):

    serializer_class = ChangePasswordSerializer
    model = Account
    permission_classes = (IsAuthenticated,)
    authentication_classes = (TokenAuthentication,)

    def get_object(self, queryset=None):
        obj = self.request.user
        return obj

    def update(self, request, *args, **kwargs):
        self.object = self.get_object()
        serializer = self.get_serializer(data=request.data)

        if serializer.is_valid():
            # Check old password
            if not self.object.check_password(serializer.data.get("old_password")):
                return Response({"old_password": ["Wrong password."]}, status=status.HTTP_400_BAD_REQUEST)

            # confirm the new passwords match
            new_password = serializer.data.get("new_password")
            confirm_new_password = serializer.data.get("confirm_new_password")
            if new_password != confirm_new_password:
                return Response({"new_password": ["New passwords must match"]}, status=status.HTTP_400_BAD_REQUEST)

            # set_password also hashes the password that the user will get
            self.object.set_password(serializer.data.get("new_password"))
            self.object.save()
            return Response({"response": "successfully changed password"}, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


################################################################

@api_view(['POST', ])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def bloodReqt(request):
    if request.method == 'POST':
        # print(request.data)
        age = int(request.data['age'])
        bmi = float(request.data['bmi'])
        glucouse = float(request.data['glucouse'])
        insuline = float(request.data['insuline'])
        homa = float(request.data['homa'])
        leptin = float(request.data['leptin'])
        adiponcetin = float(request.data['adiponcetin'])
        resistiin = float(request.data['resistiin'])
        mcp = float(request.data['mcp'])
        all_data = [age, bmi, glucouse, insuline,
                    homa, leptin, adiponcetin, resistiin, mcp]
        # ML Part
        loaded_model = joblib.load(open("savedModels/bloodmodelRBF", 'rb'))
        clf = loaded_model.predict(
            [[age, bmi, glucouse, insuline, homa, leptin, adiponcetin, resistiin, mcp]])

        for i in range(1):
            if (clf[i] == 0):
                data = "No Cancer"
            elif (clf[i] == 1):
                data = "Cancer"
        serializers = bloodTestSeializers(
            data={**request.data, 'user': request.user.id, 'result': data})
        if serializers.is_valid():
            serializers.save()
        else:
            return Response(serializers.errors)

        return Response({'blood_analysis_result': [all_data, data]})


@api_view(['GET', ])
@permission_classes([])
@authentication_classes([TokenAuthentication])
def blooddata(request, patient):
    if request.method == 'GET':
        tests = bloodTest.objects.filter(user_id=patient)
        s = bloodTestSeializers(tests, many=True)
    return Response(s.data)


@api_view(['GET', ])
@permission_classes([])
@authentication_classes([TokenAuthentication])
def blood_data_all(request):
    if request.method == 'GET':
        tests = bloodTest.objects.all()
        s = bloodTestDocterSeializers(tests, many=True)
    return Response(s.data)

###################################################################################################################################


@api_view(['POST', ])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def diabetes_result(request):

    model = pickle.load(
        open("savedModels/diabetes-prediction-rfc-model.pkl", "rb"))
    if request.method == 'POST':
        pregnancies = float(request.data['pregnancies'])
        glucose = float(request.data['glucose'])
        bloodpressure = float(request.data['bloodpressure'])
        skinthickness = float(request.data['skinthickness'])
        insulin = float(request.data['insulin'])
        bmi = float(request.data['bmi'])
        dpf = float(request.data['dpf'])
        age = int(request.data['age'])

        all_data = [pregnancies, glucose, bloodpressure,
                    skinthickness, insulin, bmi, dpf, age]
        prediction = model.predict(
            [[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]])
        if int(prediction[0]) == 1:
            data = 'diabetic'
        elif int(prediction[0]) == 0:
            data = "not diabetic"
        serializers = diabtesTestSeializers(
            data={**request.data, 'user': request.user.id, 'result': data})
        if serializers.is_valid():
            serializers.save()
        else:
            return Response(serializers.errors)

        return Response({'diabetes_result': [all_data, data]})


@api_view(['GET', ])
@permission_classes([])
@authentication_classes([TokenAuthentication])
def diabtesdata(request, patient):
    if request.method == 'GET':

        tests = diabtesTest.objects.filter(user_id=patient, null=True)
        s = diabtesTestSeializers(tests, many=True)
    return Response(s.data)


@api_view(['GET', ])
@permission_classes([])
@authentication_classes([TokenAuthentication])
def diabtes_data_all(request):
    if request.method == 'GET':
        tests = diabtesTest.objects.all()
        s = diabtesTestDocterSeializers(tests, many=True)
    return Response(s.data)
#######################################################################################################


@api_view(['POST', ])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def alzheimer_result(request):
    if request.method == 'POST':
        gender = str(request.data['gender'])
        age = int(request.data['Age'])
        EDUC = int(request.data['EDUC'])
        SES = float(request.data['SES'])
        MMSE = float(request.data['MMSE'])
        eTIV = float(request.data['eTIV'])
        nWBV = float(request.data['nWBV'])
        ASF = float(request.data['ASF'])
        if gender == 'female':
            gender = 0
        else:
            gender = 1
        all_data = [gender, age, EDUC, SES, MMSE, eTIV, nWBV, ASF]
        scaler = pickle.load(open("savedModels/alzheimer.scl", "rb"))
        model = pickle.load(
            open(r"savedModels/alzheimer.model", "rb"))
        # ML Part
        scaled_feature = scaler.transform(
            [[gender, age, EDUC, SES, MMSE, eTIV, nWBV, ASF]])
        clf = model.predict(scaled_feature)

        if (clf == 0):
            data = "Nondemented	"
        elif (clf == 1):
            data = "Demented"
        serializers = alzhimarTestSeializers(
            data={**request.data, 'user': request.user.id, 'result': data})
        if serializers.is_valid():
            serializers.save()
        else:
            return Response(serializers.errors)
        return Response({'alzheimer_result': [all_data, data]})


@api_view(['GET', ])
@permission_classes([])
@authentication_classes([TokenAuthentication])
def alzheimersdata(request, patient):
    if request.method == 'GET':
        tests = alzhimarTest.objects.filter(user_id=patient)
        s = alzhimarTestSeializers(tests, many=True)
    return Response(s.data)


@api_view(['GET', ])
@permission_classes([])
@authentication_classes([TokenAuthentication])
def alzheimer_data_all(request):
    if request.method == 'GET':
        tests = alzhimarTest.objects.all()
        s = alzhimarTestDocterSeializers(tests, many=True)
    return Response(s.data)


# 3
@api_view(['POST', ])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def heart_disease_result(request):
    df = pd.read_csv('savedModels/Heart_train (1).csv')
    df["sex"] = df["sex"] .map({"female": 1, "male": 0})
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1:]

    value = ''

    if request.method == 'POST':
        name = str(request.data['name'])
        age = float(request.data['age'])
        sex = str(request.data['sex'])
        cp = float(request.data['cp'])
        trestbps = float(request.data['trestbps'])
        chol = float(request.data['chol'])
        fbs = float(request.data['fbs'])
        restecg = float(request.data['restecg'])
        thalach = float(request.data['thalach'])
        exang = float(request.data['exang'])
        oldpeak = float(request.data['oldpeak'])
        slope = float(request.data['slope'])
        ca = float(request.data['ca'])
        thal = float(request.data['thal'])
        all_data = [name, age, sex, cp, trestbps, chol, fbs,
                    restecg, thalach, exang, oldpeak, slope, ca, thal]
        if sex == 'male':
            sex = 0
        else:
            sex = 1

        user_data = np.array(
            (age,
             sex,
             cp,
             trestbps,
             chol,
             fbs,
             restecg,
             thalach,
             exang,
             oldpeak,
             slope,
             ca,
             thal)
        ).reshape(1, 13)

        rf = RandomForestClassifier(
            n_estimators=16,
            criterion='entropy',
            max_depth=9
        )

        rf.fit(np.nan_to_num(X), Y)
        rf.score(np.nan_to_num(X), Y)
        predictions = rf.predict(user_data)

        if int(predictions[0]) == 1:
            data = 'Have a heart attack'
        elif int(predictions[0]) == 0:
            data = "don't have a heart attack"
        serializers = heartTestSeializers(
            data={**request.data, 'user': request.user.id, 'result': data})
        if serializers.is_valid():
            serializers.save()
        else:
            return Response(serializers.errors)
        return Response({'heart_disease_result': [all_data, data]})


@api_view(['GET', ])
@permission_classes([])
@authentication_classes([TokenAuthentication])
def heartdata(request, patient):
    if request.method == 'GET':
        tests = heartTest.objects.filter(user_id=patient)
        s = heartTestSeializers(tests, many=True)
    return Response(s.data)


@api_view(['GET', ])
@permission_classes([])
@authentication_classes([TokenAuthentication])
def heart_data_all(request):
    if request.method == 'GET':
        tests = heartTest.objects.all()
        s = heartTestDocterSeializers(tests, many=True)
    return Response(s.data)

#####################################################################################


@api_view(['POST', ])
@permission_classes([IsAuthenticated])
@authentication_classes([TokenAuthentication])
def parkinson_result(request):
    if request.method == 'POST':
        MDVP_Fo_Hz = float(request.data['MDVP_Fo_Hz'])
        MDVP_Fhi_Hz = float(request.data['MDVP_Fhi_Hz'])
        MDVP_Flo_Hz = float(request.data['MDVP_Flo_Hz'])
        MDVP_Jitter = float(request.data['MDVP_Jitter'])
        MDVP_Jitter_Abs = float(request.data['MDVP_Jitter_Abs'])
        MDVP_RAP = float(request.data['MDVP_RAP'])
        MDVP_PPQ = float(request.data['MDVP_PPQ'])
        Jitter_DDP = float(request.data['Jitter_DDP'])
        MDVP_Shimmer = float(request.data['MDVP_Shimmer'])
        MDVP_Shimmer_dB = float(request.data['MDVP_Shimmer_dB'])
        Shimmer_APQ3 = float(request.data['Shimmer_APQ3'])
        Shimmer_APQ5 = float(request.data['Shimmer_APQ5'])
        MDVP_APQ = float(request.data['MDVP_APQ'])
        Shimmer_DDA = float(request.data['Shimmer_DDA'])
        NHR = float(request.data['NHR'])
        HNR = float(request.data['HNR'])
        RPDE = float(request.data['RPDE'])
        DFA = float(request.data['DFA'])
        spread1 = float(request.data['spread1'])
        spread2 = float(request.data['spread2'])
        D2 = float(request.data['D2'])
        PPE = float(request.data['PPE'])
        # ML Part
        all_data = [MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer,
                    MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        model = joblib.load(
            r"D:\final project\MedicalLap\savedModels/Predict_Parkinson.model")
        feature = [[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer,
                    MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]
        data = model.predict(feature)
        if data == 0:
            data = "Uninfected"
        else:
            data = "infected"
        serializers = parkinsonTestSeializers(
            data={**request.data, 'user': request.user.id, 'result': data})
        if serializers.is_valid():
            serializers.save()
        else:
            return Response(serializers.errors)
        return Response({'parkinson_result': [all_data, data]})


@api_view(['GET', ])
@permission_classes([])
@authentication_classes([TokenAuthentication])
def parkinsonsdata(request, patient):
    if request.method == 'GET':
        tests = parkinsonTest.objects.filter(user_id=patient)
        s = parkinsonTestSeializers(tests, many=True)
    return Response(s.data)


@api_view(['GET', ])
@permission_classes([])
@authentication_classes([TokenAuthentication])
def parkinson_data_all(request):
    if request.method == 'GET':
        tests = parkinsonTest.objects.all()
        s = parkinsonTestDocterSeializers(tests, many=True)
    return Response(s.data)


# 333
class PostView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        posts = Post.objects.all()
        serializer = PostTestSerializer(posts, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        posts_serializer = PostTestSerializer(data=request.data)
        if posts_serializer.is_valid():
            post_obj = posts_serializer.save()
            print(post_obj.image.path)
            modedl_path = r"savedModels\chestExploration.hdf5"
            model = keras.models.load_model(modedl_path)
            gray_image = cv2.imread(post_obj.image.path, 0)
            resized_image = cv2.resize(gray_image, (100, 100))
            scaled_image = resized_image.astype("float32") / 255.0
        #  1 image, 100, 100 dim , 1 no of chanels
            sample_batch = scaled_image.reshape(1, 100, 100, 1)
            result = model.predict(sample_batch)
            result[result >= 0.5] = 1  # Normal
            result[result < 0.5] = 0  # Pneimonia
            if result[0][0] == 1:
                result = "Normal"
            else:
                result = "Pneimonia"

            return Response({'result': result})
            # return Response(posts_serializer.data.update(result=result), status=status.HTTP_201_CREATED)
        else:
            print('error', posts_serializer.errors)
            return Response(posts_serializer.errors, status=status.HTTP_400_BAD_REQUEST)




        
        
@api_view(['GET', ])
@permission_classes([])
@authentication_classes([TokenAuthentication])
def chestdata(request, patient):
    if request.method == 'GET':
        tests = Post.objects.filter(user_id=patient)
        s = PostTestSerializer(tests, many=True)
    return Response(s.data)

##################################################################################################