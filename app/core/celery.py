from celery import Celery
from app.core.config import settings

# Create Celery instance
celery_app = Celery(
    "digital_griot",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.tasks.transcription",
        "app.tasks.translation",
        "app.tasks.analysis"
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    broker_connection_retry_on_startup=True,
)

# Optional: Configure task routing
celery_app.conf.task_routes = {
    "app.tasks.transcription.*": {"queue": "transcription"},
    "app.tasks.translation.*": {"queue": "translation"},
    "app.tasks.analysis.*": {"queue": "analysis"},
}

if __name__ == "__main__":
    celery_app.start() 