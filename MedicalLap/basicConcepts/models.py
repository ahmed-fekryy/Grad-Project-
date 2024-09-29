from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token


class MyAccountManager(BaseUserManager):
    def create_user(self, email, username, password=None):
        if not email:
            raise ValueError('Users must have an email address')
        if not username:
            raise ValueError('Users must have a username')

        user = self.model(
            email=self.normalize_email(email),
            username=username,
        )

        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, username, password):
        user = self.create_user(
            email=self.normalize_email(email),
            password=password,
            username=username,
        )
        user.is_admin = True
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)
        return user


class Account(AbstractBaseUser):
    email = models.EmailField(verbose_name="email", max_length=60, unique=True)
    username = models.CharField(max_length=30, unique=True)
    date_joined = models.DateTimeField(
        verbose_name='date joined', auto_now_add=True)
    last_login = models.DateTimeField(verbose_name='last login', auto_now=True)
    is_admin = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)
    is_docter = models.BooleanField(default=False)
    is_patient = models.BooleanField(default=False)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    objects = MyAccountManager()

    def __str__(self):
        return self.email

    # For checking permissions. to keep it simple all admin have ALL permissons
    def has_perm(self, perm, obj=None):
        return self.is_admin

    # Does this user have permission to view this app? (ALWAYS YES FOR SIMPLICITY)
    def has_module_perms(self, app_label):
        return True


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.create(user=instance)


###########################################
# bloodmodel


class bloodTest(models.Model):
    user = models.ForeignKey(Account, on_delete=models.CASCADE)
    age = models.IntegerField()
    bmi = models.FloatField()
    glucouse = models.FloatField()
    insuline = models.FloatField()
    homa = models.FloatField()
    leptin = models.FloatField()
    adiponcetin = models.FloatField()
    resistiin = models.FloatField()
    mcp = models.FloatField()
    result = models.CharField(max_length=20)

    def __str__(self):
        return str(self.age)


##################################################

# diabtesmodel

class diabtesTest(models.Model):
    user = models.ForeignKey(
        Account, on_delete=models.CASCADE)
    pregnancies = models.FloatField()
    glucose = models.FloatField()
    bloodpressure = models.FloatField()
    skinthickness = models.FloatField()
    insulin = models.FloatField()
    bmi = models.FloatField()
    dpf = models.FloatField()
    age = models.IntegerField()
    result = models.CharField(max_length=20)

    def __str__(self):
        return str(self.pregnancies)


##########################################################

# parkinson


class parkinsonTest(models.Model):
    user = models.ForeignKey(Account, on_delete=models.CASCADE)
    MDVP_Fo_Hz = models.FloatField()
    MDVP_Fhi_Hz = models.FloatField()
    MDVP_Flo_Hz = models.FloatField()
    MDVP_Jitter = models.FloatField()
    MDVP_Jitter_Abs = models.FloatField()
    MDVP_RAP = models.FloatField()
    MDVP_PPQ = models.FloatField()
    Jitter_DDP = models.FloatField()
    MDVP_Shimmer = models.FloatField()
    MDVP_Shimmer_dB = models.FloatField()
    Shimmer_APQ3 = models.FloatField()
    Shimmer_APQ5 = models.FloatField()
    MDVP_APQ = models.FloatField()
    Shimmer_DDA = models.FloatField()
    NHR = models.FloatField()
    HNR = models.FloatField()
    RPDE = models.FloatField()
    DFA = models.FloatField()
    spread1 = models.FloatField()
    spread2 = models.FloatField()
    D2 = models.FloatField()
    PPE = models.FloatField()
    result = models.CharField(max_length=20)

    def __str__(self):
        return str(self.NHR)

#########################################################
# alzimar


class alzhimarTest(models.Model):

    gender = models.CharField(max_length=15, choices=(
        [('male', 'male'), ('female', 'female')]))
    Age = models.FloatField()
    EDUC = models.FloatField()
    SES = models.FloatField()
    MMSE = models.FloatField()
    eTIV = models.FloatField()
    nWBV = models.FloatField()
    ASF = models.FloatField()
    user = models.ForeignKey(Account, on_delete=models.CASCADE)
    result = models.CharField(max_length=20)

    def __str__(self):
        return str(self.EDUC)


# ######################################################
# heart


class heartTest(models.Model):
    user = models.ForeignKey(Account, on_delete=models.CASCADE)
    GENDER_CHOICE = (
        ('male', 'male'),
        ('female', 'female')
    )
    name = models.CharField(max_length=15)
    age = models.FloatField()
    sex = models.CharField(max_length=15, choices=GENDER_CHOICE)
    cp = models.FloatField()
    trestbps = models.FloatField()
    chol = models.FloatField()
    fbs = models.FloatField()
    restecg = models.FloatField()
    thalach = models.FloatField()
    exang = models.FloatField()
    oldpeak = models.FloatField()
    slope = models.FloatField()
    ca = models.FloatField()
    thal = models.FloatField()
    result = models.CharField(max_length=50)

    def __str__(self):
        return str(self. name)


#####################################################################
# chest


# class chestTest(models.Model):
#     user = models.ForeignKey(Account, on_delete=models.CASCADE)
#     name = models.CharField(max_length=15)
#     age = models.IntegerField()
#     file = models.ImageField(upload_to="chest_photos")
#     resultChest = models.CharField(max_length=20)

#     def __str__(self):
#         return str(self. name)


class Post(models.Model):
    # user = models.ForeignKey(Account, on_delete=models.CASCADE)
    title = models.CharField(max_length=100, null=True, blank=True)
    content = models.TextField(null=True, blank=True)
    image = models.ImageField(upload_to='post_images')
    resultPost = models.CharField(max_length=20, blank=True, null=True)

    def __str__(self):
        return self.title
