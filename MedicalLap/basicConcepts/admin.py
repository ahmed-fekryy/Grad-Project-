from django.contrib import admin

from . models import *

admin.site.register(Account)
admin.site.register(bloodTest)
admin.site.register(alzhimarTest)
admin.site.register(diabtesTest)
admin.site.register(heartTest)
admin.site.register(Post)
admin.site.register(parkinsonTest)

