# pyrefly: ignore [missing-import]
from django.contrib import admin
# pyrefly: ignore [missing-import]
from .models import Document

# Register your models here.
admin.site.register(Document)
