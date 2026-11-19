"""
ASGI config for agenticRag project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

from ai.short_term_mem import get_checkpointer

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'agenticRag.settings')

application = get_asgi_application()
get_checkpointer()
