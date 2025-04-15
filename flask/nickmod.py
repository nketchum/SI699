from datetime import date
from markupsafe import Markup

# from flaskext.markdown import Markdown

import os


def sayHello():
	today = str(date.today())
	markup = Markup(f'<code>{today}</code>')
	return markup


def makeMessage(message):
	return message


def makePlots():
	full_filename = os.path.join('/static', 'img.png') # os.getcwd() breaks this.
	markup = Markup(f'<img src="{full_filename}">')
	return markup


def makeText():
	full_filename = os.path.join(os.getcwd() + '/static', 'textfile.md')
	text = open(full_filename, 'r')
	# markdown = Markdown(text)
	return text

def run_hello():
    with open("hello.txt", "w") as f:
        f.write("Hello World")
    return "hello.txt"
