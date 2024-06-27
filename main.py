import flet as ft
from whisper_ui.whisper_app import WhisperApp


def main(page: ft.Page):
    page.title="Speech to Text AI"
    app = WhisperApp(page)
    page.add(app)
    
ft.app(target=main)


