import datetime
import json

from google.cloud import storage

from config.load_config import load_config
from logger import logger


def save_chat_history(history, session_timestamp):
    try:
        config = load_config("config/config.yaml")
        gcs_bucket_name = config.get("bucket_name")

        filename = f"{session_timestamp}_chat_history.json"
        file_path_gcs = f"chat_history/{filename}"
        json_content = json.dumps(history, ensure_ascii=False, indent=4)

        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket_name)
        blob = bucket.blob(file_path_gcs)

        blob.upload_from_string(json_content, content_type="application/json; charset=utf-8")
        logger.debug(f"chat history saved to GCS at location: gs://{gcs_bucket_name}/{file_path_gcs}")
    except Exception as e:
        logger.info(f"Error in save_chat_history: {e}")


def save_feedback(feedback_text, user_name, chat_history, session_timestamp):
    try:
        config = load_config("config/config.yaml")
        gcs_bucket_name = config.get("bucket_name")

        feedback_filename = f"{session_timestamp}_feedback.json"
        feedback_file_path_gcs = f"feedback/{feedback_filename}"

        feedback_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "user_name": user_name,
            "feedback_text": feedback_text,
            "chat_history": chat_history,
        }
        json_feedback_content = json.dumps(feedback_data, ensure_ascii=False, indent=4)

        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket_name)
        feedback_blob = bucket.blob(feedback_file_path_gcs)

        feedback_blob.upload_from_string(json_feedback_content, content_type="application/json; charset=utf-8")
        logger.debug(f"feedback saved to GCS at location: gs://{gcs_bucket_name}/{feedback_file_path_gcs}")
        return "תודה רבה על המשוב!"
    except Exception as e:
        logger.info(f"Error in save_feedback: {e}")
        return "אירעה שגיאה בשמירת המשוב. אנא נסה שוב מאוחר יותר."


def get_new_session_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
