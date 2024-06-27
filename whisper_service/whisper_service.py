from whisper_service.transcribe import transcribe as transcribe_function
from whisper_service import load_model as load_model_function
from googletrans import Translator


class TranscribeResult:
    def __init__(self, is_success: bool = False, text: str = ""):
        self.is_success = is_success
        self.text = text

def translate_text(text: str, target_language: str = "en") -> str:
    if target_language not in ["en","hi"]:
        raise ValueError("Target language must be either 'en' (English)")

    translator = Translator()
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text


def recognize(
    audio: str, model_name: str, target_languages: list = ["en", "hi"], partial_result_received=None, output_data_received=None
):
    """Returned True, if recognition process finished successfully."""
    try:
        loaded_model = load_model_function(
            model_name, output_data_receive=output_data_received
        )
        result = transcribe_function(
            model=loaded_model,
            audio=audio,
            verbose=True,
            partial_result_receive=partial_result_received,
            output_data_receive=output_data_received,
        )
        translated_texts = {}
        for language in target_languages:
            translated_texts[language] = translate_text(result["text"], language)
        
        # Prepare text for writing into the file
        file_text = ""
        for language, text in translated_texts.items():
            file_text += f"{'English' if language == 'en' else 'Hindi'}: {text}\n"
        
        return TranscribeResult(True, file_text)
    except Exception as e:
        output_data_received(str(e))
        return TranscribeResult()
