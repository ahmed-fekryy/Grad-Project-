from rest_framework import serializers
from basicConcepts.models import Account
from .models import *



class bloodTestSeializers(serializers.ModelSerializer):
    class Meta:
        model = bloodTest
        fields = '__all__'


class alzhimarTestSeializers(serializers.ModelSerializer):
    class Meta:
        model = alzhimarTest
        fields = '__all__'


class diabtesTestSeializers(serializers.ModelSerializer):
    class Meta:
        model = diabtesTest
        fields = '__all__'


class heartTestSeializers(serializers.ModelSerializer):
    class Meta:
        model = heartTest
        fields = '__all__'


class PostTestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = '__all__'


class parkinsonTestSeializers(serializers.ModelSerializer):
    class Meta:
        model = parkinsonTest
        fields = '__all__'


###############################################################################################

class doctor_serializer(serializers.ModelSerializer):
    is_docter = serializers.BooleanField(default=True)
    class Meta : 
        model = Account
        fields = '__all__'
        

class pation_serializer(serializers.ModelSerializer):
    is_patient = serializers.BooleanField(default=True)
    class Meta:
        model = Account
        fields = '__all__'

        

class RegistrationSerializer(serializers.ModelSerializer):

    password2 = serializers.CharField(
        style={'input_type': 'password'}, write_only=True)

    class Meta:
        model = Account
        fields = ['email', 'username', 'password',
                  'password2', 'is_docter', 'is_patient']
        extra_kwargs = {
            'password': {'write_only': True},
        }

    def save(self):
        account = Account( 
            email=self.validated_data['email'],
            username=self.validated_data['username'],
            is_docter=self.validated_data.get('is_docter', False),
            is_patient=self.validated_data.get('is_patient', False),
        )
        password = self.validated_data['password']
        password2 = self.validated_data['password2']
        if password != password2:
            raise serializers.ValidationError(
                {'password': 'Passwords must match.'})
        account.set_password(password)
        account.save()
        return account


class ChangePasswordSerializer(serializers.Serializer):

	old_password = serializers.CharField(required=True)
	new_password = serializers.CharField(required=True)
	confirm_new_password = serializers.CharField(required=True)

##############################################################################################


class bloodTestDocterSeializers(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='user.username')
    class Meta:
        model = bloodTest
        fields = '__all__'


class alzhimarTestDocterSeializers(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='user.username')
    class Meta:
        model = alzhimarTest
        fields = '__all__'


class diabtesTestDocterSeializers(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='user.username')
    class Meta:
        model = diabtesTest
        fields = '__all__'


class heartTestDocterSeializers(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='user.username')
    class Meta:
        model = heartTest
        fields = '__all__'


class PostTestDocterSeializers(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='user.username')
    class Meta:
        model = Post
        fields = '__all__'


class parkinsonTestDocterSeializers(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='user.username')
    class Meta:
        model = parkinsonTest
        fields = '__all__'
