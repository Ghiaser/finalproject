from celery import Celery

celery = Celery(
    'secure_clip',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0',
)
# Configure Celery to run tasks eagerly (and not attempt to send to a separate worker)
celery.conf.task_always_eager = True
celery.conf.task_eager_propagates = True

# Additional settings can be added if needed
celery.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='Asia/Jerusalem',
    enable_utc=True,
)
