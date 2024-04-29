import re

def extract_questions(text):
    text_after_colon = text.split(':', 1)[1] if ':' in text else text   
    pattern = r'(?<=\?)\s*(?=[A-Z0-9])'
    questions = re.split(pattern, text_after_colon)
    questions = [question.strip() for question in questions if question.strip().endswith('?')]
    return questions